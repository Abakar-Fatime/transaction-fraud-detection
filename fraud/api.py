"""Scoring service: send one transaction, get a risk score, a decision and reasons.

    uvicorn fraud.api:app --port 8000        # then open http://localhost:8000
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from fraud import explain
from fraud.features import ALL_FEATURES, OnlineFeatures

ROOT = Path(__file__).resolve().parent.parent


class Transaction(BaseModel):
    step: int
    type: str
    amount: float
    nameOrig: str
    oldbalanceOrg: float
    newbalanceOrig: float
    nameDest: str
    oldbalanceDest: float
    newbalanceDest: float


def create_app(model_dir: Path | None = None) -> FastAPI:
    mdir = model_dir or ROOT / "models"
    booster = lgb.Booster(model_file=str(mdir / "lgbm_all.txt"))
    meta = json.loads((mdir / "meta.json").read_text())
    state = OnlineFeatures()           # destination history, kept in memory since the service started
    app = FastAPI(title="Transaction fraud scoring")

    @app.get("/health")
    def health():
        return {"status": "ok", "features": len(meta["features"]), "threshold": meta["threshold"]}

    @app.post("/score")
    def score(tx: Transaction):
        t0 = time.perf_counter()
        feats = state.transform(tx.model_dump())
        x = pd.DataFrame([feats])[ALL_FEATURES]
        p = float(booster.predict(x)[0])
        contrib = booster.predict(x, pred_contrib=True)[0][:-1]
        reasons = explain.top_reasons(x.iloc[0], np.asarray(contrib), ALL_FEATURES, 5)
        return {"score": p, "flagged": p >= meta["threshold"], "threshold": meta["threshold"],
                "reasons": reasons, "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}

    console = ROOT / "console"
    if console.exists():
        @app.get("/")
        def index():
            return FileResponse(console / "index.html")
        app.mount("/", StaticFiles(directory=console), name="console")
    return app


try:
    app = create_app()
except Exception:      # models not trained yet; importing the module should still work
    app = FastAPI()
