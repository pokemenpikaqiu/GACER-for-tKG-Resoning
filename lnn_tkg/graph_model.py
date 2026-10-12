"""Spatio-temporal tKG predictor: CompGCN snapshots + CfC time evolution.

For each query time ``t`` we take the ``w`` most recent snapshots before ``t``:

    snapshot G_i --(CompGCN, L layers)--> H_i  (num_ent, dim)     [spatial]
    [H_{t-w} ... H_{t-1}]  --(CfC over the window)--> H*           [temporal]
    query (s, r, ?):  q = MLP([H*[s] ; rel[r]])
    logits = q @ H*^T + entity_bias                               [decode]

The CfC plays the role of RE-GCN's recurrent unit but as a continuous-time
Liquid network operating per-entity over the snapshot sequence (entities are the
batch dimension, the window is the time dimension).
"""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

_EXP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_NCPS_PATH = os.path.join(_EXP, "libs", "lnn_ncps")
if _NCPS_PATH not in sys.path:
    sys.path.insert(0, _NCPS_PATH)

from ncps.torch import CfC, LTC  # noqa: E402
from ncps.torch.cfc_cell import CfCCell  # noqa: E402


def snapshot_gaps(times):
    """Inter-snapshot elapsed time for a sorted list of time indices."""
    if len(times) <= 1:
        return [0.0] * len(times)
    gaps = [0.0]
    for i in range(1, len(times)):
        gaps.append(float(times[i] - times[i - 1]))
    return gaps


def thin_snapshots(times, edges, keep, rng):
    """Randomly drop snapshots; recompute gaps from surviving timestamps."""
    if keep >= 1.0 or len(times) == 0:
        return times, edges, snapshot_gaps(times)
    mask = rng.rand(len(times)) < keep
    if not mask.any():
        mask[rng.randint(len(times))] = True
    t_kept = [t for t, m in zip(times, mask) if m]
    e_kept = [e for e, m in zip(edges, mask) if m]
    return t_kept, e_kept, snapshot_gaps(t_kept)


class CompGCNLayer(nn.Module):
    """One relational message-passing layer over a single snapshot.

    Messages flow into the subject node ``s`` from neighbour ``o`` composed with
    the relation embedding: ``msg = W_rel(h_o + r)``; mean-aggregated per node and
    combined with a self transform. Because the snapshot already contains inverse
    edges (o, r+R, s), both directions are covered by aggregating to column 0.
    """

    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.W_self = nn.Linear(dim, dim)
        self.W_rel = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, s_idx, r_idx, o_idx, rel_emb):
        out = self.W_self(x)
        if s_idx.numel() > 0:
            msg = self.W_rel(x[o_idx] + rel_emb[r_idx])      # (E, dim)
            agg = torch.zeros_like(x)
            agg.index_add_(0, s_idx, msg)
            deg = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
            deg.index_add_(0, s_idx, torch.ones(s_idx.numel(), device=x.device, dtype=x.dtype))
            out = out + agg / deg.clamp(min=1.0).unsqueeze(1)
        return self.drop(F.relu(out))


class GraphTKGModel(nn.Module):
    def __init__(
        self,
        num_ent,
        num_rel,
        dim=200,
        gcn_layers=2,
        temporal="cfc",
        dropout=0.2,
        copy=False,
    ):
        super().__init__()
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.dim = dim
        self.temporal_name = temporal.lower()
        self.copy = copy

        self.ent_emb = nn.Embedding(num_ent, dim)
        self.rel_emb = nn.Embedding(num_rel, dim)
        self.ent_bias = nn.Parameter(torch.zeros(num_ent))

        self.gcn = nn.ModuleList(
            [CompGCNLayer(dim, dropout) for _ in range(gcn_layers)]
        )

        if self.temporal_name == "cfc_ct":
            # Native continuous-time: inter-snapshot gap drives the ODE step size.
            self.cell = CfCCell(dim, dim)
            self.temporal = None
        elif self.temporal_name == "cfc_feat":
            # Time enters as log1p(gap) concatenated to entity states each step.
            self.temporal = CfC(dim + 1, dim, return_sequences=False, batch_first=True)
            self.cell = None
        elif self.temporal_name == "cfc":
            self.temporal = CfC(dim, dim, return_sequences=False, batch_first=True)
            self.cell = None
        elif self.temporal_name == "ltc":
            self.temporal = LTC(dim, dim, return_sequences=False, batch_first=True)
            self.cell = None
        elif self.temporal_name == "gru":
            self.temporal = nn.GRU(dim, dim, batch_first=True)
            self.cell = None
        else:
            raise ValueError(f"Unknown temporal unit '{temporal}'")

        self.decoder_mlp = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

        if self.copy:
            # copy pathway projects the evolved subject state + relation, then
            # scores only against the historical (s, r) vocabulary.
            self.copy_mlp = nn.Sequential(
                nn.Linear(2 * dim, dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim, dim),
            )
            self.copy_beta = nn.Parameter(torch.ones(1))
            self.tau_raw = nn.Parameter(torch.full((num_rel,), 30.0))
            self.gate_mlp = nn.Sequential(
                nn.Linear(dim + 3, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.ent_emb.weight)
        nn.init.xavier_uniform_(self.rel_emb.weight)

    def encode_snapshot(self, edges):
        """edges: (s_idx, r_idx, o_idx). Returns H_i (num_ent, dim)."""
        x = self.ent_emb.weight
        s_idx, r_idx, o_idx = edges
        for layer in self.gcn:
            x = layer(x, s_idx, r_idx, o_idx, self.rel_emb.weight)
        return x

    def evolve(self, window_edges, snap_gaps=None):
        """window_edges: list of (s,r,o) tuples (oldest -> newest).

        ``snap_gaps`` : (L,) inter-snapshot elapsed time; required for ``cfc_ct``
        and ``cfc_feat`` (defaults to ones if omitted).
        Returns the evolved entity matrix H* (num_ent, dim).
        """
        if len(window_edges) == 0:
            return self.ent_emb.weight
        seq = torch.stack([self.encode_snapshot(e) for e in window_edges], dim=1)
        N, L, _ = seq.shape
        device = seq.device

        if self.temporal_name == "cfc_ct":
            if snap_gaps is None:
                gaps = torch.ones(L, device=device)
            else:
                gaps = torch.log1p(snap_gaps.to(device).float())
            h = torch.zeros(N, self.dim, device=device)
            for i in range(L):
                ts = gaps[i : i + 1].expand(N, 1)
                h, _ = self.cell(seq[:, i], h, ts)
            return h

        if self.temporal_name == "cfc_feat":
            if snap_gaps is None:
                gaps = torch.ones(L, device=device)
            else:
                gaps = torch.log1p(snap_gaps.to(device).float())
            gap_feat = gaps.view(1, L, 1).expand(N, L, 1)
            seq = torch.cat([seq, gap_feat], dim=-1)
            h = self.temporal(seq)
            return h[0] if isinstance(h, tuple) else h

        if self.temporal_name in ("cfc", "ltc"):
            h = self.temporal(seq)
            return h[0] if isinstance(h, tuple) else h

        _, h = self.temporal(seq)
        return h[-1]

    def decode(self, hstar, s, r, c_obj=None, c_cnt=None, c_dt=None):
        """Score candidates for queries (s, r, ?) against evolved states.

        hstar : (num_ent, dim); s, r : (B,).
        If ``copy`` is enabled, ``c_obj/c_cnt/c_dt`` (B, K) provide the historical
        vocabulary and the method returns **log-probabilities** (use NLLLoss);
        otherwise it returns raw generate logits (use CrossEntropyLoss).
        """
        feat = torch.cat([hstar[s], self.rel_emb(r)], dim=-1)
        q = self.decoder_mlp(feat)
        gen_logits = q @ hstar.t() + self.ent_bias        # (B, num_ent)

        if not self.copy:
            return gen_logits

        B, K = c_obj.shape
        valid = (c_obj >= 0).float()
        idx = c_obj.clamp(min=0)

        tau = F.softplus(self.tau_raw[r]).unsqueeze(1) + 1e-3   # (B, 1)
        cand_w = c_cnt * torch.exp(-c_dt / tau) * valid         # (B, K)

        vocab_mask = torch.zeros(B, self.num_ent, device=gen_logits.device)
        vocab_mask.scatter_add_(1, idx, valid)
        vocab_mask = (vocab_mask > 0).float()

        copy_prior = torch.zeros(B, self.num_ent, device=gen_logits.device)
        copy_prior.scatter_add_(1, idx, cand_w)
        copy_prior = torch.log1p(copy_prior)

        qc = self.copy_mlp(feat)
        copy_logits = qc @ hstar.t() + self.copy_beta * copy_prior
        copy_logits = copy_logits.masked_fill(vocab_mask == 0, -1e9)

        gen_p = torch.softmax(gen_logits, dim=-1)
        copy_p = torch.softmax(copy_logits, dim=-1)

        has_vocab = (vocab_mask.sum(dim=1, keepdim=True) > 0).float()
        total_freq = (c_cnt * valid).sum(dim=1, keepdim=True)
        distinct = valid.sum(dim=1, keepdim=True)
        masked_dt = torch.where(valid > 0, c_dt, torch.full_like(c_dt, 1e6))
        min_dt = masked_dt.min(dim=1, keepdim=True).values
        gate_stats = torch.cat(
            [torch.log1p(total_freq), torch.log1p(distinct), torch.log1p(min_dt)],
            dim=-1,
        )
        alpha = torch.sigmoid(self.gate_mlp(torch.cat([q, gate_stats], dim=-1)))
        alpha_eff = alpha * has_vocab

        p = alpha_eff * copy_p + (1.0 - alpha_eff) * gen_p
        return torch.log(p + 1e-12)
