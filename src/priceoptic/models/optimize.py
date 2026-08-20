"""Constrained profit-maximizing price.

With constant elasticity e and unit cost c, profit(P) = (P - c) * k * P^e.
The unconstrained optimum P* = c * e / (1 + e) (for e < -1); inelastic
products (e >= -1) push to the price ceiling, which is exactly why the
max-change constraint exists. We optimize on a grid inside the constraints —
transparent and robust to any demand shape swapped in later.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from priceoptic.settings import get_config


@dataclass
class PriceRec:
    current_price: float
    recommended_price: float
    change_pct: float
    expected_profit_lift_pct: float
    binding_constraint: str

    def as_dict(self) -> dict:
        return {k: round(v, 4) if isinstance(v, float) else v for k, v in vars(self).items()}


def profit(price: np.ndarray, cost: float, elasticity: float, ref_price: float) -> np.ndarray:
    demand = (price / ref_price) ** elasticity  # relative demand
    return (price - cost) * demand


def optimize_price(current_price: float, cost: float, elasticity: float) -> PriceRec:
    cfg = get_config()["optimization"]
    lo = max(current_price * (1 - cfg["max_price_change"]), cost * (1 + cfg["min_margin"]))
    hi = current_price * (1 + cfg["max_price_change"])
    if lo >= hi:
        lo = hi * 0.999

    grid = np.linspace(lo, hi, 500)
    profits = profit(grid, cost, elasticity, current_price)
    best_idx = int(np.argmax(profits))
    best_price = float(grid[best_idx])

    current_profit = float(profit(np.array([current_price]), cost, elasticity, current_price)[0])
    lift = (
        (float(profits[best_idx]) - current_profit) / abs(current_profit) if current_profit else 0.0
    )

    binding = "none"
    if best_idx == 0:
        binding = "floor (margin or max-change)"
    elif best_idx == len(grid) - 1:
        binding = "ceiling (max-change)"

    return PriceRec(
        current_price=current_price,
        recommended_price=round(best_price, 2),
        change_pct=(best_price - current_price) / current_price,
        expected_profit_lift_pct=lift,
        binding_constraint=binding,
    )
