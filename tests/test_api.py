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
