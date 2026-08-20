"""A/B revenue simulator: current vs candidate price under estimation noise.

Bootstraps the elasticity from its standard error, simulates weekly demand
noise, and reports the distribution of the revenue/profit difference — the
honest answer to "what happens if we ship this price?" including the chance
the change backfires.
"""

from __future__ import annotations

import numpy as np

from priceoptic.settings import get_config


def simulate_ab(
    current_price: float,
    candidate_price: float,
    cost: float,
    elasticity: float,
    elasticity_se: float,
    base_weekly_units: float,
    seed: int = 42,
) -> dict:
    cfg = get_config()["simulator"]
    rng = np.random.default_rng(seed)
    n, weeks = cfg["n_bootstrap"], cfg["weeks"]

    e_draws = rng.normal(elasticity, max(elasticity_se, 1e-6), n)
    profit_diffs, revenue_diffs = [], []
    for e in e_draws:
        noise_a = rng.lognormal(0, 0.18, weeks)
        noise_b = rng.lognormal(0, 0.18, weeks)
        units_a = base_weekly_units * noise_a
        units_b = base_weekly_units * (candidate_price / current_price) ** e * noise_b
        profit_a = ((current_price - cost) * units_a).sum()
        profit_b = ((candidate_price - cost) * units_b).sum()
        profit_diffs.append(profit_b - profit_a)
        revenue_diffs.append((candidate_price * units_b).sum() - (current_price * units_a).sum())

    profit_diffs = np.array(profit_diffs)
    return {
        "weeks": weeks,
        "profit_diff_mean": round(float(profit_diffs.mean()), 2),
        "profit_diff_p5": round(float(np.quantile(profit_diffs, 0.05)), 2),
        "profit_diff_p95": round(float(np.quantile(profit_diffs, 0.95)), 2),
        "prob_profit_positive": round(float((profit_diffs > 0).mean()), 4),
        "revenue_diff_mean": round(float(np.mean(revenue_diffs)), 2),
    }
