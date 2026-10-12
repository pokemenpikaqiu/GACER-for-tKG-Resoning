"""Ranking metrics for tKG link prediction: MRR and Hits@1/3/10.

Provides both *raw* ranks and *time-aware filtered* ranks. Time-aware filtering
removes all other true objects of the same (s, r, t) query from the candidate
ranking before scoring the gold entity (standard tKG evaluation protocol).
"""

import torch


class RankAccumulator:
    def __init__(self):
        self.ranks = []

    def update(self, ranks):
        # ranks: 1-D LongTensor of 1-indexed ranks
        self.ranks.append(ranks.detach().cpu())

    def result(self):
        if not self.ranks:
            return {"mrr": 0.0, "hits@1": 0.0, "hits@3": 0.0, "hits@10": 0.0, "n": 0}
        r = torch.cat(self.ranks).float()
        return {
            "mrr": (1.0 / r).mean().item(),
            "hits@1": (r <= 1).float().mean().item(),
            "hits@3": (r <= 3).float().mean().item(),
            "hits@10": (r <= 10).float().mean().item(),
            "n": int(r.numel()),
        }


def compute_ranks(logits, gold, queries=None, tf_filter=None):
    """Compute 1-indexed ranks of the gold entity.

    logits   : (B, num_ent)
    gold     : (B,) gold object ids
    queries  : optional list of (s, r, t) python tuples for filtering
    tf_filter: optional dict (s, r, t) -> set(true objects)
    """
    if tf_filter is not None and queries is not None:
        logits = logits.clone()
        for i, key in enumerate(queries):
            others = tf_filter.get(key)
            if others:
                g = int(gold[i])
                for o in others:
                    if o != g:
                        logits[i, o] = float("-inf")

    gold_scores = logits.gather(1, gold.view(-1, 1))      # (B, 1)
    # rank = 1 + number of entities scoring strictly higher than the gold
    higher = (logits > gold_scores).sum(dim=1)
    ranks = higher + 1
    return ranks
