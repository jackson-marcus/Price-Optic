"""/ingest and /live: the running estimate is what /optimize prices on."""

from __future__ import annotations


def _priceable_product(client) -> dict:
    rows = client.get("/products").json()
    return next(r for r in rows if r["priceable"])


def test_products_carry_live_estimate(client):
    rows = client.get("/products").json()
    assert {"elasticity", "se", "n_obs", "priceable"} <= set(rows[0])
    assert all(r["n_obs"] > 0 for r in rows)


def test_live_and_optimize_agree_on_seeded_history(client):
    row = _priceable_product(client)
    pid = row["product_id"]
    live = client.get(f"/live/{pid}").json()
    opt = client.get(f"/optimize/{pid}").json()
    assert live["estimate"]["elasticity"] == opt["elasticity"] == row["elasticity"]
    assert live["recommendation"]["recommended_price"] == opt["recommended_price"]
    assert opt["n_obs"] == row["n_obs"]


def test_ingest_moves_the_recommendation(client):
    """Thirty weeks of a deep discount that sells five times the baseline is
    strong evidence the product is more elastic than the history said; the
    optimiser should now want a lower price than it did before."""
    row = _priceable_product(client)
    pid = row["product_id"]
    before = client.get(f"/optimize/{pid}").json()
    weeks = [
        {
            "product_id": pid,
            "week": 104 + k,
            "price": round(row["base_price"] * 0.7, 2),
            "units": 5000,
        }
        for k in range(30)
    ]
    r = client.post("/ingest", json={"observations": weeks})
    assert r.status_code == 200
    body = r.json()
    assert (body["folded"], body["duplicates"], body["rejected"]) == (30, 0, [])
    (delta,) = body["deltas"]
    assert delta["product_id"] == pid
    assert delta["before"]["recommendation"]["recommended_price"] == before["recommended_price"]
    after = client.get(f"/optimize/{pid}").json()
    assert after["n_obs"] == before["n_obs"] + 30
    assert after["elasticity"] < before["elasticity"]
    assert after["recommended_price"] < before["recommended_price"]
    assert delta["after"]["recommendation"]["recommended_price"] == after["recommended_price"]


def test_ingest_replay_is_a_no_op(client):
    row = _priceable_product(client)
    pid = row["product_id"]
    batch = {"observations": [{"product_id": pid, "week": 300, "price": 12.5, "units": 40}]}
    first = client.post("/ingest", json=batch).json()
    again = client.post("/ingest", json=batch).json()
    assert first["folded"] == 1
    assert (again["folded"], again["duplicates"]) == (0, 1)
    assert again["deltas"] == []
    assert client.get(f"/optimize/{pid}").json()["n_obs"] == row["n_obs"] + 1


def test_ingest_reports_malformed_rows_without_dropping_good_ones(client):
    pid = _priceable_product(client)["product_id"]
    r = client.post(
        "/ingest",
        json={
            "observations": [
                {"product_id": pid, "week": 400, "price": 12.5, "units": 40},
                {"product_id": pid, "week": 401, "price": -3, "units": 40},
                {"product_id": pid, "week": 402, "units": 40},
            ]
        },
    )
    body = r.json()
    assert body["folded"] == 1
    assert [x["reason"][:8] for x in body["rejected"]] == ["price mu", "missing "]


def test_ingest_unknown_product_is_refused_whole(client):
    n_before = client.get("/live/1").json()["estimate"]["n_obs"]
    r = client.post(
        "/ingest",
        json={
            "observations": [
                {"product_id": 1, "week": 500, "price": 12.5, "units": 40},
                {"product_id": 999, "week": 500, "price": 12.5, "units": 40},
            ]
        },
    )
    assert r.status_code == 404
    assert client.get("/live/1").json()["estimate"]["n_obs"] == n_before, "good row must not land"
