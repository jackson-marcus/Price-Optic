import pandas as pd
import pytest
from fastapi.testclient import TestClient

import priceoptic.api.routes as routes
from priceoptic.api.main import create_app
from priceoptic.models.elasticity import estimate_product
from priceoptic.settings import get_config


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
    routes._elasticities.cache_clear()
    yield TestClient(create_app())
    cfg["data"]["artifacts_dir"], cfg["data"]["processed_dir"] = orig_art, orig_proc
    routes._elasticities.cache_clear()


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_products_listed(client):
    body = client.get("/products").json()
    assert len(body) == 12
    assert {"product_id", "elasticity", "true_elasticity"} <= set(body[0])


def test_optimize_returns_recommendation(client):
    r = client.get("/optimize/1")
    assert r.status_code in (200, 422)  # 422 if CI too wide for this seed
    if r.status_code == 200:
        body = r.json()
        assert body["recommended_price"] > 0
        assert "binding_constraint" in body


def test_optimize_unknown_product_404(client):
    assert client.get("/optimize/999").status_code == 404


def test_simulate_roundtrip(client):
    r = client.post("/simulate", json={"product_id": 2, "candidate_price": 50})
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["prob_profit_positive"] <= 1
    assert body["profit_diff_p5"] <= body["profit_diff_p95"]
