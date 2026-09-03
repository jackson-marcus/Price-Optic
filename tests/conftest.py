"""Fixtures: synthetic sales with known elasticities."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import priceoptic.api.routes as routes
from priceoptic.api.main import create_app
from priceoptic.models.elasticity import estimate_product
from priceoptic.settings import get_config

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from make_sales import generate


@pytest.fixture(scope="session")
def market():
    return generate(n_products=12, n_weeks=104, seed=7)


@pytest.fixture()
def client(market, tmp_path):
    products, sales = market
    cfg = get_config()
    orig_art, orig_proc = cfg["data"]["artifacts_dir"], cfg["data"]["processed_dir"]
    art, proc = tmp_path / "artifacts", tmp_path / "processed"
    art.mkdir()
    proc.mkdir()
    cfg["data"]["artifacts_dir"], cfg["data"]["processed_dir"] = str(art), str(proc)

    rows = []
    for _, product in products.iterrows():
        est = estimate_product(sales[sales["product_id"] == product["product_id"]])
        rows.append({"product_id": product["product_id"], **est})
    products.merge(pd.DataFrame(rows), on="product_id").to_parquet(
        art / "elasticities.parquet", index=False
    )
    sales.to_parquet(proc / "sales.parquet", index=False)
    routes.reset_live_state()
    yield TestClient(create_app())
    cfg["data"]["artifacts_dir"], cfg["data"]["processed_dir"] = orig_art, orig_proc
    routes.reset_live_state()
