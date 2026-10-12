"""Unified data loader for temporal knowledge graph (tKG) extrapolation.

Handles both event-based timestamp formats found in this project:
  * integer timestamp index  -> e.g. ICEWS18 / GDELT  ("s r o t 0")
  * ISO date string          -> e.g. ICEWS14 / ICEWS05-15  ("s r o 2014-05-13 ...")

The loader maps the raw (sortable) timestamps onto a dense, temporally ordered
integer index 0..T-1, adds inverse relations so that subject prediction reduces
to object prediction, and builds per-entity event histories that are queried
with binary search at train/eval time.
"""

import os
import datetime
import numpy as np


def _parse_time(token):
    """Parse a raw timestamp token into a sortable integer.

    Returns None if the token cannot be parsed (e.g. interval placeholders
    like ``####-##-##`` used by YAGO11k / WIKI12k).
    """
    token = token.strip()
    if token.lstrip("-").isdigit():
        return int(token)
    # ISO date, possibly with '##' placeholders for unknown month/day.
    parts = token.split("-")
    if len(parts) >= 3:
        year, month, day = parts[0], parts[1], parts[2]
        if "#" in year:
            return None
        month = "01" if "#" in month else month
        day = "01" if "#" in day else day
        try:
            return datetime.date(int(year), int(month), int(day)).toordinal()
        except ValueError:
            return None
    return None


def _read_quads(path):
    """Read a split file -> list of (s, r, o, raw_time)."""
    quads = []
    with open(path, "r") as fr:
        for line in fr:
            cols = line.split()
            if len(cols) < 4:
                continue
            s, r, o = int(cols[0]), int(cols[1]), int(cols[2])
            t = _parse_time(cols[3])
            if t is None:
                continue
            quads.append((s, r, o, t))
    return quads


def get_total_number(stat_path):
    with open(stat_path, "r") as fr:
        line = fr.readline().split()
        return int(line[0]), int(line[1])


class TKGData:
    """Loads a dataset directory and exposes splits + history index.

    Attributes
    ----------
    num_ent : int
    num_rel : int                 number of *original* relations
    num_rel_aug : int             == 2 * num_rel  (with inverse relations)
    train / valid / test : np.ndarray of shape (N, 4) == (s, r_aug, o, t_idx)
    """

    def __init__(self, root, name, add_inverse=True):
        ddir = os.path.join(root, name)
        self.name = name
        self.num_ent, self.num_rel = get_total_number(os.path.join(ddir, "stat.txt"))

        raw = {
            "train": _read_quads(os.path.join(ddir, "train.txt")),
            "valid": _read_quads(os.path.join(ddir, "valid.txt")),
            "test": _read_quads(os.path.join(ddir, "test.txt")),
        }

        # Densify timestamps onto a contiguous, temporally-ordered index.
        all_times = sorted({q[3] for s in raw.values() for q in s})
        self.time2idx = {t: i for i, t in enumerate(all_times)}
        self.num_time = len(all_times)

        self.add_inverse = add_inverse
        self.num_rel_aug = self.num_rel * 2 if add_inverse else self.num_rel

        self.train = self._to_array(raw["train"])
        self.valid = self._to_array(raw["valid"])
        self.test = self._to_array(raw["test"])

        # "Known facts" available for building history at each split (extrapolation:
        # only facts strictly earlier than the query time are ever used).
        self._train_hist = self._build_history(self.train)
        self._valid_hist = self._build_history(np.concatenate([self.train, self.valid]))
        self._test_hist = self._build_history(
            np.concatenate([self.train, self.valid, self.test])
        )

        # Time-aware filter: (s, r, t) -> set of true objects (union of all splits).
        # Used ONLY at evaluation time for filtered ranking.
        self.tf_filter = self._build_filter(
            np.concatenate([self.train, self.valid, self.test])
        )
        # Train-only labels: (s, r, t) -> set of true objects from the training
        # split. Used to build 1-vs-N multi-label targets WITHOUT leaking
        # valid/test facts into training.
        self.train_filter = self._build_filter(self.train)

    def _to_array(self, quads):
        rows = []
        for s, r, o, t in quads:
            ti = self.time2idx[t]
            rows.append((s, r, o, ti))
            if self.add_inverse:
                rows.append((o, r + self.num_rel, s, ti))
        return np.asarray(rows, dtype=np.int64)

    def _build_history(self, facts):
        """For each entity build time-sorted arrays of (time, rel, neighbour).

        Returns dict: ent -> (times[np.int64], rels[np.int64], objs[np.int64]),
        each sorted ascending by time.
        """
        order = np.argsort(facts[:, 3], kind="stable")
        facts = facts[order]
        hist = {}
        # group rows by subject entity
        subj = facts[:, 0]
        sort_idx = np.argsort(subj, kind="stable")
        facts_s = facts[sort_idx]
        subj_s = facts_s[:, 0]
        uniq, starts = np.unique(subj_s, return_index=True)
        starts = list(starts) + [len(facts_s)]
        for k, e in enumerate(uniq):
            block = facts_s[starts[k] : starts[k + 1]]
            # block already globally time-sorted because we time-sorted first and
            # used stable sort on subject -> within-subject order is by time.
            hist[int(e)] = (
                block[:, 3].astype(np.int64),
                block[:, 1].astype(np.int64),
                block[:, 2].astype(np.int64),
            )
        return hist

    def _build_filter(self, facts):
        filt = {}
        for s, r, o, t in facts:
            filt.setdefault((int(s), int(r), int(t)), set()).add(int(o))
        return filt

    def history_for(self, split):
        return {
            "train": self._train_hist,
            "valid": self._valid_hist,
            "test": self._test_hist,
        }[split]
