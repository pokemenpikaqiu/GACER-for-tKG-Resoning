"""Snapshot / timeline structures for RE-GCN-style graph evolution.

Builds, on top of :class:`data.TKGData`, the per-timestamp knowledge-graph
snapshots used by the spatio-temporal model:

  * ``snap[ti]`` -> LongTensor (E, 3) of augmented edges (s, r, o) at time ``ti``
  * windowed access: the ``w`` most recent *non-empty* snapshots strictly before
    a query time (extrapolation: never peek at the present or future).

Three "known" timelines are exposed, mirroring the extrapolation protocol:
  * ``train``  : snapshots from the training split only
  * ``valid``  : snapshots from train + valid
  * ``test``   : snapshots from train + valid + test

Each is used to provide *history* for queries in the corresponding split, while
the query timestamp's own snapshot is excluded from its window.

When ``copy_max > 0`` each timeline also precomputes, for every query fact, the
CyGNet-style copy vocabulary over the entities historically observed with the
same (s, r) pair strictly before the query time: candidate objects ``c_obj``,
their occurrence counts ``c_cnt`` and the gap ``c_dt`` to the most recent
occurrence (the model applies a learnable per-relation recency decay).
"""

import bisect

import numpy as np
import torch


def _group_indices_by_time(t_col):
    """Return dict t -> ndarray of row indices (stable, time-grouped)."""
    order = np.argsort(t_col, kind="stable")
    sorted_t = t_col[order]
    uniq, starts = np.unique(sorted_t, return_index=True)
    starts = list(starts) + [len(order)]
    return {int(ti): order[starts[k] : starts[k + 1]] for k, ti in enumerate(uniq)}


def _compute_copy(query_facts, history, copy_max, copy_tau):
    """Per-fact copy vocabulary aligned to ``query_facts`` row order.

    Returns (c_obj, c_cnt, c_dt), each (N, K) with K=copy_max; padding = -1 / 0.
    """
    n = len(query_facts)
    K = copy_max
    c_obj = np.full((n, K), -1, dtype=np.int64)
    c_cnt = np.zeros((n, K), dtype=np.float32)
    c_dt = np.zeros((n, K), dtype=np.float32)
    for i in range(n):
        s, r, o, t = (
            int(query_facts[i, 0]),
            int(query_facts[i, 1]),
            int(query_facts[i, 2]),
            int(query_facts[i, 3]),
        )
        h = history.get(s)
        if h is None:
            continue
        times, rels, objs = h
        end = np.searchsorted(times, t, side="left")  # strictly before t
        if end == 0:
            continue
        same_r = rels[:end] == r
        if not same_r.any():
            continue
        cand = objs[:end][same_r]
        cand_t = times[:end][same_r]
        uniq, inv = np.unique(cand, return_inverse=True)
        cnt = np.zeros(uniq.size, dtype=np.float32)
        np.add.at(cnt, inv, 1.0)
        recent_t = np.zeros(uniq.size, dtype=np.float32)
        np.maximum.at(recent_t, inv, cand_t.astype(np.float32))
        dt = (t - recent_t).astype(np.float32)
        score = cnt * np.exp(-dt / copy_tau)  # selection only
        top = np.argsort(-score)[:K]
        m = top.size
        c_obj[i, :m] = uniq[top]
        c_cnt[i, :m] = cnt[top]
        c_dt[i, :m] = dt[top]
    return c_obj, c_cnt, c_dt


class Timeline:
    """A set of snapshots with windowed access and per-time query facts."""

    def __init__(self, snap_facts, query_facts, num_time, history=None,
                 copy_max=0, copy_tau=30.0):
        self.num_time = num_time
        self.copy_max = copy_max
        self._edge_cache = {}

        # snapshots used to build history (grouped edges)
        snap_groups = _group_indices_by_time(snap_facts[:, 3])
        self.snaps = {
            ti: snap_facts[idx][:, [0, 1, 2]].astype(np.int64)
            for ti, idx in snap_groups.items()
        }
        self.snap_times = sorted(self.snaps.keys())

        # query facts grouped by time (the facts we actually predict)
        if copy_max > 0:
            c_obj, c_cnt, c_dt = _compute_copy(query_facts, history, copy_max, copy_tau)
        q_groups = _group_indices_by_time(query_facts[:, 3])
        self.queries = {}
        self.copy = {}
        for ti, idx in q_groups.items():
            self.queries[ti] = query_facts[idx][:, [0, 1, 2]].astype(np.int64)
            if copy_max > 0:
                self.copy[ti] = (c_obj[idx], c_cnt[idx], c_dt[idx])
        self.query_times = sorted(self.queries.keys())

    def edges(self, ti, device):
        if ti not in self._edge_cache:
            self._edge_cache[ti] = torch.from_numpy(self.snaps[ti])
        e = self._edge_cache[ti].to(device)
        return e[:, 0], e[:, 1], e[:, 2]

    def copy_tensors(self, ti, device):
        c_obj, c_cnt, c_dt = self.copy[ti]
        return (
            torch.from_numpy(c_obj).to(device),
            torch.from_numpy(c_cnt).to(device),
            torch.from_numpy(c_dt).to(device),
        )

    def window(self, ti, w):
        """The ``w`` most recent snapshot times strictly before ``ti``."""
        pos = bisect.bisect_left(self.snap_times, ti)
        return self.snap_times[max(0, pos - w) : pos]


def build_timelines(data, copy_max=0, copy_tau=30.0):
    """Return dict split -> Timeline using the extrapolation history rule."""
    train = data.train
    valid = data.valid
    test = data.test
    tv = np.concatenate([train, valid])
    tvt = np.concatenate([train, valid, test])
    return {
        "train": Timeline(train, train, data.num_time, data.history_for("train"),
                          copy_max, copy_tau),
        "valid": Timeline(tv, valid, data.num_time, data.history_for("valid"),
                          copy_max, copy_tau),
        "test": Timeline(tvt, test, data.num_time, data.history_for("test"),
                         copy_max, copy_tau),
    }
