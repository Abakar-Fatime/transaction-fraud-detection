import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from fraud import models, pipeline
from fraud.data import make_fixture
from fraud.features import ALL_FEATURES, HARD_FEATURES, build


def test_drain_rule_flags_account_emptying_transfers():
    df = make_fixture(3000, seed=6)
    X = build(df)
    flagged = models.drain_rule(X) == 1
    assert flagged[df.isFraud.to_numpy() == 1].mean() > 0.95
    assert flagged[df.isFraud.to_numpy() == 0].mean() < 0.02


def test_evasion_simulation_only_touches_test_window_fraud():
    df = make_fixture(4000, seed=7)
    ev = pipeline.evade(df)
    changed = (ev["amount"] != df["amount"]) | (ev["newbalanceOrig"] != df["newbalanceOrig"])
    expected = (df.isFraud == 1) & (df.step > 37)
    assert (changed == expected).all()
    f = ev[expected]
    assert ((f.amount / f.oldbalanceOrg).round(2) == 0.8).all()
    assert (f.oldbalanceOrg - f.amount - f.newbalanceOrig).abs().max() < 0.02


def test_rule_catches_less_after_evasion():
    df = make_fixture(6000, seed=8, fraud_rate=0.04)
    X = build(df)
    ev_df = pipeline.evade(df)
    assert models.drain_rule(build(ev_df)).sum() < models.drain_rule(X).sum()


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    d = tmp_path_factory.mktemp("run")
    csv = d / "fixture.csv"
    make_fixture(8000, seed=9, fraud_rate=0.04).to_csv(csv, index=False)
    res = pipeline.run(str(csv), out_dir=str(d / "results"), console_dir=str(d / "console"),
                       model_dir=str(d / "models"), top_alerts=20, quick=True)
    return d, res


def test_pipeline_writes_every_artifact(trained):
    d, res = trained
    for f in ["results/metrics.json", "console/console.json", "models/lgbm_all.txt", "models/meta.json"]:
        assert (d / f).exists()
    names = [m["model"] for m in res["models"]]
    assert "LightGBM, all features" in names and "Drain rule (hand-written)" in names


def test_console_export_is_consistent(trained):
    d, _ = trained
    c = json.loads((d / "console/console.json").read_text())
    assert len(c["alerts"]) == 20
    scores = [a["score"] for a in c["alerts"]]
    assert scores == sorted(scores, reverse=True)
    assert all(len(a["reasons"]) == 5 for a in c["alerts"])
    assert c["curve"]["caught"] == sorted(c["curve"]["caught"])
    assert max(c["curve"]["caught"]) <= c["curve"]["total_fraud"]


def test_api_scores_and_explains(trained):
    from fraud.api import create_app
    d, _ = trained
    client = TestClient(create_app(d / "models"))
    assert client.get("/health").json()["status"] == "ok"
    legit = dict(step=40, type="PAYMENT", amount=50.0, nameOrig="C1", oldbalanceOrg=1000.0,
                 newbalanceOrig=950.0, nameDest="M7", oldbalanceDest=0.0, newbalanceDest=0.0)
    drain = dict(step=40, type="TRANSFER", amount=5000.0, nameOrig="C2", oldbalanceOrg=5000.0,
                 newbalanceOrig=0.0, nameDest="C77", oldbalanceDest=0.0, newbalanceDest=0.0)
    a, b = client.post("/score", json=legit).json(), client.post("/score", json=drain).json()
    assert b["score"] > a["score"]
    assert len(b["reasons"]) == 5 and "latency_ms" in b
    assert client.post("/score", json={"step": 1}).status_code == 422
