"""API routes: /products, /optimize/{product_id}, /simulate, /ingest, /live/{product_id}, /health.

Elasticities served here come from the live feed, not straight from the
parquet artifact: on first use the sales history is replayed through the
ledger into ``ElasticityUpdater``, which then keeps moving as ``/ingest``
receives new weeks. The artifact still supplies product metadata and the
synthetic ground truth.
"""

from __future__ import annotations

import functools
import json
import logging
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from priceoptic.models.optimize import optimize_price
from priceoptic.settings import get_config, resolve_path
from priceoptic.simulator.ab import simulate_ab
from priceoptic.streams.consumer import LedgerConsumer
from priceoptic.streams.producer import SalesLedger
from priceoptic.workers.processor import ElasticityUpdater, LiveEstimate, review_delta

logger = logging.getLogger(__name__)
router = APIRouter()

SE_CEILING = 0.5  # refuse to price on an interval wider than this


class SimRequest(BaseModel):
    product_id: int = Field(ge=1)
    candidate_price: float = Field(gt=0)


class IngestRequest(BaseModel):
    observations: list[dict[str, Any]] = Field(min_length=1, max_length=10_000)


@functools.lru_cache(maxsize=1)
def _elasticities() -> pd.DataFrame:
    path = resolve_path(get_config()["data"]["artifacts_dir"]) / "elasticities.parquet"
    if not path.exists():
        raise FileNotFoundError("No elasticities; run `python -m priceoptic.models.elasticity`")
    return pd.read_parquet(path)


def _sales_history() -> pd.DataFrame:
    return pd.read_parquet(resolve_path(get_config()["data"]["processed_dir"]) / "sales.parquet")


@functools.lru_cache(maxsize=1)
def _live() -> LedgerConsumer:
    """Ledger + updater seeded from the sales history. Cached for the process lifetime."""
    ledger = SalesLedger()
    ledger.publish_history(_sales_history())
    consumer = LedgerConsumer(ledger, ElasticityUpdater())
    report = consumer.drain()
    logger.info(
        "live feed seeded: %d weeks folded across %d products", report.folded, len(report.touched)
    )
    return consumer


def reset_live_state() -> None:
    _elasticities.cache_clear()
    _live.cache_clear()


def _sales_baseline(product_id: int) -> float:
    sales = _sales_history()
    recent = sales[sales["product_id"] == product_id].tail(12)
    return float(recent["units"].mean()) if len(recent) else 100.0


def _product(product_id: int) -> pd.Series:
    df = _elasticities()
    match = df[df["product_id"] == product_id]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"Unknown product_id {product_id}")
    return match.iloc[0]


def _recommend(product: pd.Series, est: LiveEstimate) -> dict | None:
    """Optimiser output for a live estimate, or None when the estimate is not priceable."""
    if not est.priceable:
        return None
    rec = optimize_price(float(product["base_price"]), float(product["cost"]), est.elasticity)
    return rec.as_dict()


def _snapshot(product: pd.Series, est: LiveEstimate) -> dict:
    return {"estimate": est.as_dict(), "recommendation": _recommend(product, est)}


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/products")
def products() -> list[dict]:
    try:
        df = _elasticities()
        updater = _live().updater
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    cols = ["product_id", "category", "base_price", "cost", "true_elasticity"]
    rows = json.loads(df[cols].to_json(orient="records"))
    for row in rows:
        est = updater.estimate(int(row["product_id"])).as_dict()
        row.update(
            {k: est[k] for k in ("elasticity", "ci_lo", "ci_hi", "se", "n_obs", "priceable")}
        )
    return rows


@router.get("/live/{product_id}")
def live(product_id: int) -> dict:
    """Current running estimate and the recommendation it implies."""
    try:
        product = _product(product_id)
        est = _live().updater.estimate(product_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"product_id": product_id, **_snapshot(product, est)}


@router.get("/optimize/{product_id}")
def optimize(product_id: int) -> dict:
    try:
        product = _product(product_id)
        est = _live().updater.estimate(product_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if est.elasticity is None:
        raise HTTPException(
            status_code=422,
            detail=f"Elasticity not estimable for this product ({est.note}; {est.n_obs} weeks, {est.n_prices} distinct prices)",
        )
    if est.se > SE_CEILING:
        raise HTTPException(
            status_code=422,
            detail=f"Elasticity CI too wide (se={est.se:.2f}) — collect more price variation first",
        )
    rec = optimize_price(float(product["base_price"]), float(product["cost"]), est.elasticity)
    return {
        "product_id": product_id,
        "elasticity": round(est.elasticity, 4),
        "n_obs": est.n_obs,
        **rec.as_dict(),
    }


@router.post("/ingest")
def ingest(request: IngestRequest) -> dict:
    """Append new weekly sales and report what they did to each touched
    product's recommendation. Malformed rows are returned under ``rejected``;
    a (product, week) already on the ledger is counted as a duplicate and
    ignored, so replaying a batch cannot double-count a week."""
    try:
        df = _elasticities()
        consumer = _live()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    known = set(df["product_id"].astype(int))
    unknown = sorted({int(o.get("product_id", -1)) for o in request.observations} - known - {-1})
    if unknown:
        raise HTTPException(status_code=404, detail=f"Unknown product_id(s) {unknown}")

    touched = {int(o["product_id"]) for o in request.observations if "product_id" in o}
    before = {pid: _snapshot(_product(pid), consumer.updater.estimate(pid)) for pid in touched}
    _, rejects = consumer.ledger.publish(request.observations)
    report = consumer.drain()

    deltas = [
        review_delta(pid, before[pid], _snapshot(_product(pid), consumer.updater.estimate(pid)))
        for pid in sorted(report.touched)
    ]
    return {
        "folded": report.folded,
        "duplicates": report.duplicates,
        "zero_sales": report.zero_sales,
        "rejected": [{"payload": r.payload, "reason": r.reason} for r in rejects],
        "ledger_offset": consumer.offset,
        "deltas": [d.as_dict() for d in deltas],
    }


@router.post("/simulate")
def simulate(request: SimRequest) -> dict:
    try:
        product = _product(request.product_id)
        est = _live().updater.estimate(request.product_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if est.elasticity is None:
        raise HTTPException(status_code=422, detail=f"Elasticity not estimable ({est.note})")
    result = simulate_ab(
        current_price=float(product["base_price"]),
        candidate_price=request.candidate_price,
        cost=float(product["cost"]),
        elasticity=est.elasticity,
        elasticity_se=est.se,
        base_weekly_units=_sales_baseline(request.product_id),
    )
    return {"product_id": request.product_id, "candidate_price": request.candidate_price, **result}
