"""API routes: /products, /optimize/{product_id}, /simulate, /health."""

from __future__ import annotations

import functools
import json
import logging

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from priceoptic.models.optimize import optimize_price
from priceoptic.settings import get_config, resolve_path
from priceoptic.simulator.ab import simulate_ab

logger = logging.getLogger(__name__)
router = APIRouter()


class SimRequest(BaseModel):
    product_id: int = Field(ge=1)
    candidate_price: float = Field(gt=0)


@functools.lru_cache(maxsize=1)
def _elasticities() -> pd.DataFrame:
    path = resolve_path(get_config()["data"]["artifacts_dir"]) / "elasticities.parquet"
    if not path.exists():
        raise FileNotFoundError("No elasticities; run `python -m priceoptic.models.elasticity`")
    return pd.read_parquet(path)


def _sales_baseline(product_id: int) -> float:
    sales = pd.read_parquet(resolve_path(get_config()["data"]["processed_dir"]) / "sales.parquet")
    recent = sales[sales["product_id"] == product_id].tail(12)
    return float(recent["units"].mean()) if len(recent) else 100.0


def _product(product_id: int) -> pd.Series:
    df = _elasticities()
    match = df[df["product_id"] == product_id]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"Unknown product_id {product_id}")
    return match.iloc[0]


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/products")
def products() -> list[dict]:
    try:
        df = _elasticities()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    cols = [
        "product_id",
        "category",
        "base_price",
        "cost",
        "elasticity",
        "ci_lo",
        "ci_hi",
        "true_elasticity",
    ]
    return json.loads(df[cols].to_json(orient="records"))


@router.get("/optimize/{product_id}")
def optimize(product_id: int) -> dict:
    try:
        product = _product(product_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if product["elasticity"] is None or pd.isna(product["elasticity"]):
        raise HTTPException(status_code=422, detail="Elasticity not estimable for this product")
    if product["se"] > 0.5:
        raise HTTPException(
            status_code=422,
            detail=f"Elasticity CI too wide (se={product['se']:.2f}) — collect more price variation first",
        )
    rec = optimize_price(
        float(product["base_price"]), float(product["cost"]), float(product["elasticity"])
    )
    return {"product_id": product_id, "elasticity": float(product["elasticity"]), **rec.as_dict()}


@router.post("/simulate")
def simulate(request: SimRequest) -> dict:
    try:
        product = _product(request.product_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    result = simulate_ab(
        current_price=float(product["base_price"]),
        candidate_price=request.candidate_price,
        cost=float(product["cost"]),
        elasticity=float(product["elasticity"]),
        elasticity_se=float(product["se"]),
        base_weekly_units=_sales_baseline(request.product_id),
    )
    return {"product_id": request.product_id, "candidate_price": request.candidate_price, **result}
