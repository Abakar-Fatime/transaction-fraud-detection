"""End to end: load PaySim, build features, train, evaluate, explain, export for the console.

    python -m fraud run --data data/raw/PS_20174392719_1491204439457_log.csv
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from fraud import evaluate as ev
from fraud import explain, models
from fraud.data import load_paysim, split_by_time, TRAIN_END, VALID_END
from fraud.features import ALL_FEATURES, HARD_FEATURES, build

EVASION_KEEP = 0.8   # simulated fraudster leaves 20% of the balance behind


def evade(df: pd.DataFrame, keep: float = EVASION_KEEP) -> pd.DataFrame:
    """Crude evasion simulation on test-window fraud: move `keep` of the balance instead of all of it."""
    d = df.copy()
    m = (d["isFraud"] == 1) & (d["step"] > VALID_END)
    old = d.loc[m, "oldbalanceOrg"]
    amt_old = d.loc[m, "amount"]
    amt_new = np.where(old > 0, old * keep, amt_old * keep).round(2)
    d.loc[m, "newbalanceDest"] = d.loc[m, "newbalanceDest"] - amt_old + amt_new
    d.loc[m, "amount"] = amt_new
    d.loc[m, "newbalanceOrig"] = (old - amt_new).where(old > 0, 0.0)
    return d


def run(data_path: str, out_dir: str = "results", console_dir: str = "console/data",
        model_dir: str = "models", top_alerts: int = 300, quick: bool = False) -> dict:
    t0 = time.time()
    df = load_paysim(data_path)
    X = build(df)
    y = df["isFraud"].to_numpy()
    idx = {k: v.index.to_numpy() for k, v in split_by_time(df).items()}
    tr, va, te = idx["train"], idx["valid"], idx["test"]
    info = {
        "rows": int(len(df)), "steps": [int(df.step.min()), int(df.step.max())],
        "fraud": {k: int(y[v].sum()) for k, v in idx.items()}, "n": {k: int(len(v)) for k, v in idx.items()},
        "split": {"train_steps": [1, TRAIN_END], "valid_steps": [TRAIN_END + 1, VALID_END],
                  "test_steps": [VALID_END + 1, int(df.step.max())]},
        "review_cost": ev.REVIEW_COST,
    }

    # ---- fit
    rule_score = lambda X_: models.drain_rule(X_)
    lr_all = models.fit_logreg(X.iloc[tr], y[tr], ALL_FEATURES)
    lr_hard = models.fit_logreg(X.iloc[tr], y[tr], HARD_FEATURES)
    gb_all = models.fit_lgbm(X.iloc[tr], y[tr], X.iloc[va], y[va], ALL_FEATURES)
    gb_hard = models.fit_lgbm(X.iloc[tr], y[tr], X.iloc[va], y[va], HARD_FEATURES)
    iso = models.fit_iforest(X.iloc[tr], HARD_FEATURES)

    configs = {
        "Drain rule (hand-written)": (lambda Xs: rule_score(Xs), None),
        "Logistic regression, all features": (lambda Xs: models.score(lr_all, Xs, ALL_FEATURES, "logreg"), None),
        "Logistic regression, no drain features": (lambda Xs: models.score(lr_hard, Xs, HARD_FEATURES, "logreg"), None),
        "LightGBM, all features": (lambda Xs: models.score(gb_all, Xs, ALL_FEATURES, "lgbm"), None),
        "LightGBM, no drain features": (lambda Xs: models.score(gb_hard, Xs, HARD_FEATURES, "lgbm"), None),
        "Isolation Forest (unsupervised)": (lambda Xs: models.score(iso, Xs, HARD_FEATURES, "iforest"), None),
    }

    # ---- evaluate on validation and test (time-ordered, never shuffled)
    rows, scores_te = [], {}
    df_ev = evade(df)
    X_ev = build(df_ev)
    for name, (fn, _) in configs.items():
        s_va, s_te = fn(X.iloc[va]), fn(X.iloc[te])
        scores_te[name] = s_te
        s_ev = fn(X_ev.iloc[te])
        r = {"model": name, "valid": ev.summarise(y[va], s_va), "test": ev.summarise(y[te], s_te)}
        r["test_evasion"] = ev.summarise(y[te], s_ev)
        # threshold picked on validation cost, applied unchanged to test
        k = ev.best_k(y[va], s_va, df.loc[va, "amount"], ev.REVIEW_COST)
        thr = float(np.sort(s_va)[::-1][k - 1]) if k > 0 else float("inf")
        if 0 < k and thr <= 0:      # binary rule: ties at 0 would flag everything, so flag only positives
            thr = float(s_va[s_va > 0].min()) if (s_va > 0).any() else float("inf")
        r["cost"] = {"threshold": thr, **ev.cost_at_threshold(y[te], s_te, df.loc[te, "amount"], thr)}
        r["cost_evasion"] = ev.cost_at_threshold(y[te], s_ev, df.loc[te, "amount"], thr)
        if name in ("LightGBM, all features", "Drain rule (hand-written)", "LightGBM, no drain features"):
            r["test"]["pr_auc_ci95"] = list(ev.bootstrap_pr_auc(y[te], s_te, n=100 if quick else 300))
        rows.append(r)
        print(f"{name:42s} test PR-AUC {r['test']['pr_auc']:.4f}  evasion PR-AUC {r['test_evasion']['pr_auc']:.4f}", flush=True)

    # ---- explain the main model
    main = "LightGBM, all features"
    s_main = scores_te[main]
    order = np.argsort(-s_main, kind="stable")
    top = order[:top_alerts]
    Xte = X.iloc[te].reset_index(drop=True)
    dte = df.iloc[te].reset_index(drop=True)
    sv_top = explain.shap_values(gb_all, Xte.iloc[top][ALL_FEATURES])
    rng = np.random.default_rng(0)
    samp = rng.choice(len(Xte), size=min(4000, len(Xte)), replace=False)
    sv_samp = explain.shap_values(gb_all, Xte.iloc[samp][ALL_FEATURES])
    importance = sorted(({"feature": f, "mean_abs_shap": float(np.abs(sv_samp[:, i]).mean())}
                         for i, f in enumerate(ALL_FEATURES)), key=lambda d: -d["mean_abs_shap"])

    alerts = []
    for rank, (i, contrib) in enumerate(zip(top, sv_top), start=1):
        r = dte.iloc[i]
        alerts.append({
            "rank": rank, "id": int(te[i]), "step": int(r.step), "type": r.type, "amount": float(r.amount),
            "nameOrig": r.nameOrig, "nameDest": r.nameDest,
            "oldbalanceOrg": float(r.oldbalanceOrg), "newbalanceOrig": float(r.newbalanceOrig),
            "oldbalanceDest": float(r.oldbalanceDest), "newbalanceDest": float(r.newbalanceDest),
            "score": float(s_main[i]), "is_fraud": int(r.isFraud),
            "reasons": explain.top_reasons(Xte.iloc[i], contrib, ALL_FEATURES, 5),
        })

    # ---- curve for the console: for each k alerts, what is caught and what it costs
    cc = ev.cost_curve(y[te], s_main, df.loc[te, "amount"].to_numpy())
    ks = np.unique(np.concatenate([np.arange(1, 401), np.geomspace(400, len(te), 160).astype(int)]))
    cc = cc.set_index("k").loc[ks].reset_index()
    tp_cum = np.cumsum(y[te][order])
    curve = {"k": ks.tolist(), "caught": tp_cum[ks - 1].astype(int).tolist(),
             "caught_amount": cc["caught_amount"].round(2).tolist(), "score": cc["score"].round(6).tolist(),
             "total_fraud": int(y[te].sum()), "total_fraud_amount": float((y[te] * df.loc[te, "amount"]).sum()),
             "n": int(len(te))}

    # ---- drift: PSI train vs test, and score by step
    # `hour` is cyclical and the file covers under two days, so train-vs-test PSI on it only reflects the window
    check = [f for f in ALL_FEATURES if X[f].nunique() > 2 and f != "hour"]
    drift = sorted(({"feature": f, "psi": ev.psi(X.iloc[tr][f], X.iloc[te][f])} for f in check),
                   key=lambda d: -d["psi"])
    by_step = []
    for s, g in pd.DataFrame({"step": dte["step"], "s": s_main, "y": dte["isFraud"]}).groupby("step"):
        by_step.append({"step": int(s), "mean_score": float(g.s.mean()), "p99_score": float(np.quantile(g.s, 0.99)),
                        "fraud": int(g.y.sum()), "n": int(len(g))})
    base = {f: float(X.iloc[tr][f].mean()) for f in ALL_FEATURES}

    result = {"data": info, "models": rows, "importance": importance, "drift": drift,
              "evasion": {"keep": EVASION_KEEP}, "seconds": round(time.time() - t0, 1)}
    out, cdir, mdir = Path(out_dir), Path(console_dir), Path(model_dir)
    for p in (out, cdir, mdir):
        p.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    thr_main = next(r["cost"]["threshold"] for r in rows if r["model"] == main)
    (cdir / "console.json").write_text(json.dumps({
        "data": info, "models": rows, "curve": curve, "alerts": alerts, "importance": importance,
        "drift": drift, "score_by_step": by_step, "review_cost": ev.REVIEW_COST, "threshold": thr_main,
    }))
    gb_all.booster_.save_model(str(mdir / "lgbm_all.txt"))
    (mdir / "meta.json").write_text(json.dumps({"features": ALL_FEATURES, "threshold": thr_main,
                                                "review_cost": ev.REVIEW_COST, "base_means": base}))
    print(f"done in {time.time()-t0:.0f}s", flush=True)
    return result
