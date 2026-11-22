"""Loading, time-based splitting and a small PaySim-shaped fixture for tests."""
from __future__ import annotations

import numpy as np
import pandas as pd

COLUMNS = ["step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
           "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud"]

TRAIN_END = 30     # steps 1-30   train
VALID_END = 37     # steps 31-37  validation (model selection, threshold)
                   # steps 38+    test (touched once, for reporting)


def load_paysim(path: str) -> pd.DataFrame:
    """Read the Kaggle PaySim CSV. Some copies start with a stray title line, so find the header."""
    with open(path, "r", encoding="utf-8") as fh:
        skip = 0 if fh.readline().startswith("step,") else 1
    df = pd.read_csv(path, skiprows=skip)
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"not a PaySim file, missing columns: {sorted(missing)}")
    return df.reset_index(drop=True)


def split_by_time(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Train on the past, validate on the next window, test on the last. Never shuffled."""
    return {
        "train": df[df.step <= TRAIN_END],
        "valid": df[(df.step > TRAIN_END) & (df.step <= VALID_END)],
        "test": df[df.step > VALID_END],
    }


def make_fixture(n: int = 4000, seed: int = 0, fraud_rate: float = 0.03) -> pd.DataFrame:
    """A tiny PaySim-shaped dataset (same columns, same fraud signature) for tests and demos."""
    rng = np.random.default_rng(seed)
    step = np.sort(rng.integers(1, 46, n))
    typ = rng.choice(["PAYMENT", "CASH_IN", "CASH_OUT", "TRANSFER", "DEBIT"], n,
                     p=[0.34, 0.22, 0.36, 0.06, 0.02])
    bal = np.round(rng.lognormal(10.5, 1.2, n), 2)
    amt = np.round(np.minimum(bal * rng.uniform(0.01, 0.5, n), 5e5), 2)
    fraud = (np.isin(typ, ["TRANSFER", "CASH_OUT"]) & (rng.random(n) < fraud_rate * 5)).astype(int)
    amt = np.where(fraud == 1, bal, amt)
    new_o = np.where(fraud == 1, 0.0, np.round(bal - amt, 2))
    merchant = typ == "PAYMENT"
    dest = np.where(merchant, [f"M{x}" for x in rng.integers(1, 300, n)],
                    [f"C{x}" for x in rng.integers(1, 700, n)])
    old_d = np.where(merchant, 0.0, np.round(rng.lognormal(9, 1.5, n) * (rng.random(n) > 0.2), 2))
    old_d = np.where(fraud == 1, np.where(rng.random(n) < 0.6, 0.0, old_d), old_d)
    new_d = np.where(merchant, 0.0, np.round(old_d + amt, 2))
    return pd.DataFrame({
        "step": step, "type": typ, "amount": amt, "nameOrig": [f"C{i}" for i in range(10**6, 10**6 + n)],
        "oldbalanceOrg": bal, "newbalanceOrig": new_o, "nameDest": dest,
        "oldbalanceDest": old_d, "newbalanceDest": new_d, "isFraud": fraud, "isFlaggedFraud": 0,
    })
