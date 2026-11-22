"""Rule baseline, logistic regression, gradient boosting and an unsupervised detector."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
import lightgbm as lgb

from fraud.features import ALL_FEATURES, HARD_FEATURES

MONEY = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
         "orig_balance_error", "dest_balance_error", "dest_prior_amount_sum"]


def drain_rule(X: pd.DataFrame) -> np.ndarray:
    """What a hand-written rule does on this data: a transfer or cash-out that empties the sender."""
    big = (X["is_transfer"] + X["is_cash_out"]) > 0
    return (big & (X["orig_zero_balance"] == 0) & ((X["drain_ratio"] - 1).abs() <= 1e-3)).astype(float).to_numpy()


def _signed_log(a):
    a = np.asarray(a, float)
    return np.sign(a) * np.log1p(np.abs(a))


def fit_logreg(X: pd.DataFrame, y, features: list[str]):
    idx = [features.index(c) for c in MONEY if c in features]

    def squash(a):
        a = np.array(a, float, copy=True)
        a[:, idx] = _signed_log(a[:, idx])
        return a

    m = make_pipeline(FunctionTransformer(squash), StandardScaler(),
                      LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000))
    m.fit(X[features].to_numpy(float), y)
    return m


def fit_lgbm(Xtr, ytr, Xva, yva, features: list[str], seed: int = 0):
    m = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=15, min_child_samples=20,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                           reg_lambda=1.0, random_state=seed, n_jobs=2, verbose=-1)
    m.fit(Xtr[features], ytr, eval_set=[(Xva[features], yva)], eval_metric="average_precision",
          callbacks=[lgb.early_stopping(50, verbose=False)])
    return m


def fit_iforest(Xtr: pd.DataFrame, features: list[str], seed: int = 0):
    """Unsupervised: never sees labels."""
    m = IsolationForest(n_estimators=200, contamination="auto", random_state=seed, n_jobs=2)
    m.fit(Xtr[features].to_numpy(float))
    return m


def score(model, X: pd.DataFrame, features: list[str], kind: str) -> np.ndarray:
    if kind == "lgbm":
        return model.predict_proba(X[features])[:, 1]
    if kind == "logreg":
        return model.predict_proba(X[features].to_numpy(float))[:, 1]
    if kind == "iforest":
        return -model.score_samples(X[features].to_numpy(float))
    raise ValueError(kind)
