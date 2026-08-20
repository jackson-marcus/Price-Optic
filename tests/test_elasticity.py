"""Elasticity recovery + optimization + simulator sanity."""

import numpy as np

from priceoptic.models.elasticity import estimate_product
from priceoptic.models.optimize import optimize_price, profit
from priceoptic.simulator.ab import simulate_ab


def test_elasticity_recovered_within_tolerance(market):
    products, sales = market
    errors = []
    for _, product in products.iterrows():
        est = estimate_product(sales[sales["product_id"] == product["product_id"]])
        if est["elasticity"] is not None:
            errors.append(abs(est["elasticity"] - product["true_elasticity"]))
    assert len(errors) >= 10
    mae = float(np.mean(errors))
    assert mae < 0.35, f"elasticity MAE {mae:.2f} too high"
    assert mae > 0.0, "exact recovery would mean no demand noise"


def test_ci_covers_truth_mostly(market):
    products, sales = market
    covered = total = 0
    for _, product in products.iterrows():
        est = estimate_product(sales[sales["product_id"] == product["product_id"]])
        if est["elasticity"] is None:
            continue
        total += 1
        if est["ci_lo"] <= product["true_elasticity"] <= est["ci_hi"]:
            covered += 1
    assert covered / total >= 0.6, f"CI coverage {covered}/{total} too low"


def test_optimizer_moves_elastic_down_inelastic_up():
    # Elastic product (e=-2.5): margin-heavy price cut wins.
    elastic = optimize_price(current_price=100, cost=50, elasticity=-2.5)
    # Inelastic (e=-0.5): raise to the ceiling.
    inelastic = optimize_price(current_price=100, cost=50, elasticity=-0.5)
    assert elastic.recommended_price < 100
    assert inelastic.recommended_price > 100
    assert "ceiling" in inelastic.binding_constraint


def test_optimizer_respects_margin_floor():
    rec = optimize_price(current_price=10, cost=9.5, elasticity=-3.0)
    assert rec.recommended_price >= 9.5 * 1.05 - 1e-6


def test_unconstrained_optimum_matches_theory():
    # P* = c*e/(1+e) for e<-1; with wide-open constraints the grid should land near it.
    from priceoptic.settings import get_config

    cfg = get_config()["optimization"]
    original = cfg["max_price_change"]
    cfg["max_price_change"] = 5.0
    try:
        e, c = -2.0, 50.0
        rec = optimize_price(current_price=100, cost=c, elasticity=e)
        theory = c * e / (1 + e)  # = 100 for these numbers
        assert abs(rec.recommended_price - theory) / theory < 0.02
    finally:
        cfg["max_price_change"] = original


def test_profit_function_shape():
    prices = np.linspace(60, 140, 100)
    p = profit(prices, cost=50, elasticity=-2.0, ref_price=100)
    assert p.max() > 0
    assert np.argmax(p) not in (0, len(prices) - 1)


def test_simulator_prefers_theory_optimal_price():
    sim_good = simulate_ab(100, 100, 50, -2.0, 0.05, 200, seed=1)  # at optimum
    sim_bad = simulate_ab(100, 140, 50, -2.0, 0.05, 200, seed=1)  # overpriced
    assert sim_good["prob_profit_positive"] >= 0.4  # ~even at optimum vs itself
    assert sim_bad["profit_diff_mean"] < sim_good["profit_diff_mean"] + 1e-9


def test_simulator_uncertainty_widens_with_se():
    tight = simulate_ab(100, 90, 50, -2.0, 0.02, 200, seed=2)
    loose = simulate_ab(100, 90, 50, -2.0, 0.60, 200, seed=2)
    tight_width = tight["profit_diff_p95"] - tight["profit_diff_p5"]
    loose_width = loose["profit_diff_p95"] - loose["profit_diff_p5"]
    assert loose_width > tight_width
