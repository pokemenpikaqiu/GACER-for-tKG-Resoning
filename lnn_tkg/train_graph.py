"""Train / evaluate the spatio-temporal graph model (CompGCN + CfC) on tKG.

Unlike ``train.py`` (per-query history + DataLoader), this script processes the
timeline *sequentially*: for every query timestamp it builds the evolved entity
matrix H* from the ``w`` most recent snapshots and decodes all facts at that
timestamp at once. Gradients flow through the window (bounded w-step BPTT).

Example
-------
    python3 experiments/lnn_tkg/train_graph.py --dataset ICEWS14 --temporal cfc --epochs 20
"""

import os
import sys
import time
import math
import argparse

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from data import TKGData
from graph_data import build_timelines
from graph_model import GraphTKGModel, snapshot_gaps, thin_snapshots
from metrics import RankAccumulator, compute_ranks


def pick_device(arg):
    if arg != "auto":
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def window_bundle(timeline, ti, w, device, keep=1.0, rng=None, no_gap_fold=False):
    """Return (edges_list, gap_tensor) for the ``w`` snapshots before ``ti``."""
    times = timeline.window(ti, w)
    edges = [timeline.edges(t, device) for t in times]
    gaps = snapshot_gaps(times)
    if keep < 1.0 and len(times) > 0:
        rng = rng if rng is not None else np.random
        times, edges, gaps = thin_snapshots(times, edges, keep, rng)
    if no_gap_fold and keep < 1.0:
        gaps = [1.0] * len(gaps)
    gap_t = torch.tensor(gaps, dtype=torch.float32, device=device) if gaps else None
    return edges, gap_t


@torch.no_grad()
def evaluate(model, timeline, w, device, tf_filter, copy=False, eval_keep=1.0,
             eval_seed=1234, no_gap_fold=False):
    model.eval()
    flt = RankAccumulator()
    rng = np.random.RandomState(eval_seed)
    needs_gaps = model.temporal_name in ("cfc_ct", "cfc_feat")
    for ti in timeline.query_times:
        edges, gaps = window_bundle(
            timeline, ti, w, device, keep=eval_keep, rng=rng, no_gap_fold=no_gap_fold,
        )
        hstar = model.evolve(edges, snap_gaps=gaps if needs_gaps else None)
        facts = torch.from_numpy(timeline.queries[ti]).to(device)
        s, r, o = facts[:, 0], facts[:, 1], facts[:, 2]
        if copy:
            c_obj, c_cnt, c_dt = timeline.copy_tensors(ti, device)
            logits = model.decode(hstar, s, r, c_obj, c_cnt, c_dt)
        else:
            logits = model.decode(hstar, s, r)
        # Rank/filter on CPU: the per-element ``logits[i, o] = -inf`` filtering
        # trips an MPS slicing assertion on non-contiguous (transpose-based) logits.
        logits = logits.cpu()
        o = o.cpu()
        queries = list(zip(s.cpu().tolist(), r.cpu().tolist(), [ti] * s.numel()))
        flt.update(compute_ranks(logits, o, queries=queries, tf_filter=tf_filter))
    return flt.result()


def fmt(m):
    return (
        f"MRR {m['mrr']:.4f} | H@1 {m['hits@1']:.4f} | "
        f"H@3 {m['hits@3']:.4f} | H@10 {m['hits@10']:.4f} (n={m['n']})"
    )


def save_checkpoint(path, model, args, data, metric, epoch):
    ckpt = {
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "config": {
            "dataset": args.dataset,
            "model": "graph",
            "temporal": args.temporal,
            "dim": args.dim,
            "gcn_layers": args.gcn_layers,
            "window": args.window,
            "dropout": args.dropout,
            "copy": args.copy,
            "copy_max": args.copy_max,
            "copy_tau": args.copy_tau,
            "train_keep": args.train_keep,
            "no_gap_fold": args.no_gap_fold,
            "num_ent": data.num_ent,
            "num_rel": data.num_rel,
            "num_rel_aug": data.num_rel_aug,
            "num_time": data.num_time,
        },
        "time2idx": data.time2idx,
        "valid_metric": metric,
        "epoch": epoch,
    }
    torch.save(ckpt, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets"))
    p.add_argument("--dataset", default="ICEWS14")
    p.add_argument("--temporal", default="cfc_ct",
                   choices=["cfc", "cfc_ct", "cfc_feat", "ltc", "gru"])
    p.add_argument("--dim", type=int, default=300)
    p.add_argument("--gcn-layers", type=int, default=2)
    p.add_argument("--window", type=int, default=5, help="# recent snapshots in BPTT window")
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--device", default="auto")
    p.add_argument("--valid-every", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--scheduler", default="cosine", choices=["none", "cosine"])
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--copy", action="store_true", help="enable gated copy decoder")
    p.add_argument("--copy-max", type=int, default=256, help="max (s,r) copy vocabulary size")
    p.add_argument("--copy-tau", type=float, default=30.0, help="recency time-constant for copy selection")
    p.add_argument("--train-keep", type=float, default=1.0,
                   help="keep fraction of snapshots during training (1.0=off)")
    p.add_argument("--no-gap-fold", action="store_true",
                   help="ablation: do not recompute gaps after snapshot thinning")
    p.add_argument(
        "--save-path",
        default="",
        help="explicit checkpoint path (default: checkpoints/<ds>_graph_<temporal>.pt)",
    )
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--max-times", type=int, default=0, help="limit # train timestamps (smoke test)")
    args = p.parse_args()

    if args.save_path:
        ckpt_path = args.save_path
    else:
        ckpt_path = os.path.join(
            os.path.dirname(__file__), "checkpoints", f"{args.dataset}_graph_{args.temporal}.pt"
        )
    if not args.no_save:
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device(args.device)
    print(f"[setup] device={device} temporal={args.temporal} dataset={args.dataset}")

    t0 = time.time()
    data = TKGData(args.data_root, args.dataset, add_inverse=True)
    tl = build_timelines(
        data,
        copy_max=args.copy_max if args.copy else 0,
        copy_tau=args.copy_tau,
    )
    print(
        f"[data] ents={data.num_ent} rels={data.num_rel} (aug={data.num_rel_aug}) "
        f"times={data.num_time} | train={len(data.train)} valid={len(data.valid)} "
        f"test={len(data.test)} | snapshots(train)={len(tl['train'].snap_times)} "
        f"| loaded in {time.time()-t0:.1f}s"
    )

    model = GraphTKGModel(
        num_ent=data.num_ent,
        num_rel=data.num_rel_aug,
        dim=args.dim,
        gcn_layers=args.gcn_layers,
        temporal=args.temporal,
        dropout=args.dropout,
        copy=args.copy,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] graph+{args.temporal} params={n_params/1e6:.2f}M window={args.window}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = None
    if args.scheduler == "cosine":
        def lr_lambda(ep):
            if ep < args.warmup:
                return (ep + 1) / max(1, args.warmup)
            prog = (ep - args.warmup) / max(1, args.epochs - args.warmup)
            return 0.5 * (1.0 + math.cos(math.pi * prog))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    ls = args.label_smoothing
    if args.copy:
        nll = nn.NLLLoss()
        def loss_fn(out, o):                # out = log-probabilities
            loss = nll(out, o)
            if ls > 0:
                loss = (1.0 - ls) * loss + ls * (-out.mean())
            return loss
        print(f"[loss] gated copy+generate mixture, NLLLoss (label_smoothing={ls})")
    else:
        ce = nn.CrossEntropyLoss(label_smoothing=ls)
        def loss_fn(out, o):                # out = logits
            return ce(out, o)
    train_tl = tl["train"]
    train_times = list(train_tl.query_times)
    if args.max_times:
        train_times = train_times[: args.max_times]
    needs_gaps = model.temporal_name in ("cfc_ct", "cfc_feat")
    train_rng = np.random.RandomState(args.seed + 999)

    best_mrr = -1.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        np.random.shuffle(train_times)
        tot, n = 0.0, 0
        te = time.time()
        for ti in train_times:
            edges, gaps = window_bundle(
                train_tl, ti, args.window, device,
                keep=args.train_keep, rng=train_rng, no_gap_fold=args.no_gap_fold,
            )
            hstar = model.evolve(edges, snap_gaps=gaps if needs_gaps else None)
            facts = torch.from_numpy(train_tl.queries[ti]).to(device)
            s, r, o = facts[:, 0], facts[:, 1], facts[:, 2]
            if args.copy:
                c_obj, c_cnt, c_dt = train_tl.copy_tensors(ti, device)
                out = model.decode(hstar, s, r, c_obj, c_cnt, c_dt)
            else:
                out = model.decode(hstar, s, r)
            loss = loss_fn(out, o)
            opt.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            tot += loss.item() * o.size(0)
            n += o.size(0)
        if sched is not None:
            sched.step()
        msg = f"[epoch {epoch:3d}] loss {tot/max(n,1):.4f} ({time.time()-te:.1f}s)"

        if epoch % args.valid_every == 0 or epoch == args.epochs:
            flt = evaluate(model, tl["valid"], args.window, device, data.tf_filter, args.copy)
            msg += f" | valid(flt) {fmt(flt)}"
            if flt["mrr"] > best_mrr:
                best_mrr = flt["mrr"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                msg += "  *"
                if not args.no_save:
                    save_checkpoint(ckpt_path, model, args, data, flt, epoch)
        print(msg, flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    flt = evaluate(model, tl["test"], args.window, device, data.tf_filter, args.copy)
    print("=" * 70)
    print(f"[TEST filtered] {fmt(flt)}")
    print("=" * 70)

    if not args.no_save:
        ckpt = torch.load(ckpt_path, map_location="cpu") if os.path.exists(ckpt_path) else {}
        if best_state is not None:
            ckpt["model_state"] = {k: v.detach().cpu() for k, v in best_state.items()}
        ckpt["test_metric"] = {"filtered": flt}
        torch.save(ckpt, ckpt_path)
        print(f"[save] best checkpoint -> {ckpt_path}")


if __name__ == "__main__":
    main()
