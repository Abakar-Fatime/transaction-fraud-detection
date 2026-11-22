"""Metrics that make sense at ~0.05% fraud: ranking quality, an alert budget and money."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

REVIEW_COST = 100.0       # cost of one analyst review, in the dataset's currency units
BUDGETS = (0.0005, 0.001, 0.005)   # share of transactions an analyst team can review


def pr_auc(y, score) -> float:
    return float(average_precision_score(y, score))


def alert_budget(y, score, share: float) -> dict:
    """Flag the top `share` of transactions by score; report what that catches."""
    y = np.asarray(y); score = np.asarray(score)
    k = max(1, int(round(share * len(y))))
    top = np.argsort(-score, kind="stable")[:k]
    tp = int(y[top].sum())
    return {"alerts": k, "caught": tp, "recall": tp / max(1, int(y.sum())), "precision": tp / k}


def cost_curve(y, score, amount, review_cost: float = REVIEW_COST) -> pd.DataFrame:
    """Total cost when the top-k alerts are reviewed, for every k: missed fraud value + review cost."""
    y = np.asarray(y); amount = np.asarray(amount, float)
    order = np.argsort(-np.asarray(score), kind="stable")
    caught = np.cumsum(y[order] * amount[order])
    k = np.arange(1, len(y) + 1)
    total_fraud = float((y * amount).sum())
    cost = (total_fraud - caught) + review_cost * k
    return pd.DataFrame({"k": k, "caught_amount": caught, "cost": cost, "score": np.asarray(score)[order]})


def best_k(y, score, amount, review_cost: float = REVIEW_COST) -> int:
    c = cost_curve(y, score, amount, review_cost)
    best = int(c["cost"].to_numpy().argmin())
    # also allow "review nothing" (cost = all fraud lost) which is k = 0
    return 0 if float((np.asarray(y) * np.asarray(amount)).sum()) <= c["cost"].iloc[best] else best + 1


def cost_at_threshold(y, score, amount, threshold: float, review_cost: float = REVIEW_COST) -> dict:
    y = np.asarray(y); amount = np.asarray(amount, float); flag = np.asarray(score) >= threshold
    missed = float((y * amount)[~flag].sum())
    return {"alerts": int(flag.sum()), "caught": int(y[flag].sum()), "missed_amount": missed,
            "review_cost": review_cost * int(flag.sum()), "total_cost": missed + review_cost * int(flag.sum()),
            "cost_without_model": float((y * amount).sum())}


def bootstrap_pr_auc(y, score, n: int = 300, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    y = np.asarray(y); score = np.asarray(score)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum() == 0:
            continue
        vals.append(average_precision_score(y[i], score[i]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def summarise(y, score) -> dict:
    out = {"pr_auc": pr_auc(y, score), "roc_auc": float(roc_auc_score(y, score))}
    for b in BUDGETS:
        r = alert_budget(y, score, b)
        out[f"recall@{b*100:g}%"] = r["recall"]
        out[f"precision@{b*100:g}%"] = r["precision"]
    return out


def psi(expected, actual, bins: int = 10) -> float:
    """Population Stability Index of `actual` against `expected` (quantile bins from `expected`)."""
    e = np.asarray(expected, float); a = np.asarray(actual, float)
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    pe = np.histogram(e, edges)[0] / len(e)
    pa = np.histogram(a, edges)[0] / len(a)
    pe, pa = np.clip(pe, 1e-4, None), np.clip(pa, 1e-4, None)
    return float(np.sum((pa - pe) * np.log(pa / pe)))
