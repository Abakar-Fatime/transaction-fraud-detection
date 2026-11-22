"""Point-in-time features.

Every feature of a transaction is computed from that transaction plus *earlier* transactions only.
The same logic exists twice: vectorised for training (`build`) and one-event-at-a-time for serving
(`OnlineFeatures`). A test checks the two agree, so the model sees in production exactly what it
was trained on.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

TYPES = ["PAYMENT", "CASH_IN", "CASH_OUT", "TRANSFER", "DEBIT"]

# Features that expose the simulator's "empty the account" signature. Removing them is the honest
# stress test of what a model learns beyond it (see README).
DRAIN_FEATURES = ["drain_ratio", "orig_balance_error"]

ALL_FEATURES = [
    "amount", "log_amount", "hour",
    "oldbalanceOrg", "newbalanceOrig", "drain_ratio", "orig_balance_error", "orig_zero_balance",
    "oldbalanceDest", "newbalanceDest", "dest_balance_error", "dest_is_merchant",
    "dest_prior_count", "dest_prior_amount_sum", "dest_first_seen",
    "is_payment", "is_cash_in", "is_cash_out", "is_transfer", "is_debit",
]
HARD_FEATURES = [f for f in ALL_FEATURES if f not in DRAIN_FEATURES]


def _row_features(typ: str, step: int, amount: float, old_o: float, new_o: float,
                  dest: str, old_d: float, new_d: float, prior_count: int, prior_sum: float) -> dict:
    drain = amount / old_o if old_o > 0 else -1.0
    out = {
        "amount": amount,
        "log_amount": math.log1p(amount),
        "hour": step % 24,
        "oldbalanceOrg": old_o,
        "newbalanceOrig": new_o,
        "drain_ratio": drain,
        "orig_balance_error": old_o - amount - new_o,
        "orig_zero_balance": int(old_o == 0),
        "oldbalanceDest": old_d,
        "newbalanceDest": new_d,
        "dest_balance_error": old_d + amount - new_d,
        "dest_is_merchant": int(dest.startswith("M")),
        "dest_prior_count": prior_count,
        "dest_prior_amount_sum": prior_sum,
        "dest_first_seen": int(prior_count == 0),
    }
    for t in TYPES:
        out[f"is_{t.lower()}"] = int(typ == t)
    return out


def build(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorised features for a time-ordered frame. Row order defines 'earlier'."""
    d = df.reset_index(drop=True)
    g = d.groupby("nameDest", sort=False)
    prior_count = g.cumcount()
    prior_sum = g["amount"].cumsum() - d["amount"]
    x = pd.DataFrame(index=d.index)
    x["amount"] = d["amount"]
    x["log_amount"] = np.log1p(d["amount"])
    x["hour"] = d["step"] % 24
    x["oldbalanceOrg"] = d["oldbalanceOrg"]
    x["newbalanceOrig"] = d["newbalanceOrig"]
    x["drain_ratio"] = np.where(d["oldbalanceOrg"] > 0, d["amount"] / d["oldbalanceOrg"].where(d["oldbalanceOrg"] > 0, 1), -1.0)
    x["orig_balance_error"] = d["oldbalanceOrg"] - d["amount"] - d["newbalanceOrig"]
    x["orig_zero_balance"] = (d["oldbalanceOrg"] == 0).astype(int)
    x["oldbalanceDest"] = d["oldbalanceDest"]
    x["newbalanceDest"] = d["newbalanceDest"]
    x["dest_balance_error"] = d["oldbalanceDest"] + d["amount"] - d["newbalanceDest"]
    x["dest_is_merchant"] = d["nameDest"].str.startswith("M").astype(int)
    x["dest_prior_count"] = prior_count
    x["dest_prior_amount_sum"] = prior_sum
    x["dest_first_seen"] = (prior_count == 0).astype(int)
    for t in TYPES:
        x[f"is_{t.lower()}"] = (d["type"] == t).astype(int)
    return x[ALL_FEATURES]


class OnlineFeatures:
    """Streaming version: keeps per-destination state and scores one event at a time."""

    def __init__(self) -> None:
        self.count: dict[str, int] = {}
        self.total: dict[str, float] = {}

    def transform(self, tx: dict, update: bool = True) -> dict:
        dest = tx["nameDest"]
        c, s = self.count.get(dest, 0), self.total.get(dest, 0.0)
        feats = _row_features(tx["type"], int(tx["step"]), float(tx["amount"]),
                              float(tx["oldbalanceOrg"]), float(tx["newbalanceOrig"]), dest,
                              float(tx["oldbalanceDest"]), float(tx["newbalanceDest"]), c, s)
        if update:
            self.count[dest] = c + 1
            self.total[dest] = s + float(tx["amount"])
        return feats

    def warm(self, history: pd.DataFrame) -> None:
        """Load destination state from past transactions without scoring them."""
        for dest, amt in zip(history["nameDest"], history["amount"]):
            self.count[dest] = self.count.get(dest, 0) + 1
            self.total[dest] = self.total.get(dest, 0.0) + float(amt)
