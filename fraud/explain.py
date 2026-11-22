"""Plain-language reasons for a score, from SHAP values."""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap

PHRASES = {
    "drain_ratio": lambda v: ("the sender had no balance on record" if v < 0 else f"the transfer was {v*100:.0f}% of the sender's balance"),
    "orig_balance_error": lambda v: ("sender balance changed exactly by the amount" if abs(v) < 0.01 else f"sender balance does not add up (off by {v:,.2f})"),
    "orig_zero_balance": lambda v: ("sender balance was 0 before the transaction" if v else "sender had a balance before the transaction"),
    "newbalanceOrig": lambda v: (f"sender is left with {v:,.2f}" if v else "sender is left with 0"),
    "oldbalanceOrg": lambda v: f"sender balance before was {v:,.2f}",
    "amount": lambda v: f"the amount is {v:,.2f}",
    "log_amount": lambda v: f"the amount is {np.expm1(v):,.2f}",
    "oldbalanceDest": lambda v: ("the receiving account was empty before" if v == 0 else f"the receiving account held {v:,.2f} before"),
    "newbalanceDest": lambda v: f"the receiving account holds {v:,.2f} after",
    "dest_balance_error": lambda v: ("receiver balance did not change as expected" if abs(v) >= 0.01 else "receiver balance changed by the amount"),
    "dest_is_merchant": lambda v: ("the receiver is a merchant" if v else "the receiver is a personal account"),
    "dest_prior_count": lambda v: f"the receiving account has {int(v)} earlier transactions",
    "dest_prior_amount_sum": lambda v: f"the receiving account has received {v:,.2f} before",
    "dest_first_seen": lambda v: ("first time this receiving account is seen" if v else "the receiving account was seen before"),
    "hour": lambda v: f"it happened at hour {int(v)} of the day",
    "is_transfer": lambda v: ("it is a transfer" if v else "it is not a transfer"),
    "is_cash_out": lambda v: ("it is a cash-out" if v else "it is not a cash-out"),
    "is_payment": lambda v: ("it is a merchant payment" if v else "it is not a merchant payment"),
    "is_cash_in": lambda v: ("it is a cash-in" if v else "it is not a cash-in"),
    "is_debit": lambda v: ("it is a debit" if v else "it is not a debit"),
}


def phrase(feature: str, value: float) -> str:
    return PHRASES.get(feature, lambda v: f"{feature} = {v}")(value)


def shap_values(model, X: pd.DataFrame) -> np.ndarray:
    sv = shap.TreeExplainer(model).shap_values(X)
    if isinstance(sv, list):          # older shap: one array per class
        sv = sv[1]
    sv = np.asarray(sv)
    return sv[..., 1] if sv.ndim == 3 else sv


def top_reasons(row: pd.Series, contrib: np.ndarray, features: list[str], k: int = 5) -> list[dict]:
    order = np.argsort(-np.abs(contrib))[:k]
    return [{"feature": features[i], "value": float(row[features[i]]), "shap": float(contrib[i]),
             "text": phrase(features[i], float(row[features[i]]))} for i in order]
