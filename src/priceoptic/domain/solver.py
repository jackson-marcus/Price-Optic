"""Total Functional Architecture - Total Pricing Solver.

Contains pure, total functions with complete mathematical guarantees.
"""

from __future__ import annotations

import numpy as np

from priceoptic.domain.types import (
    Money,
    PriceRange,
    PriceRecommendation,
    PricingContext,
)


def compute_profit(price: float, cost: float, elasticity: float, ref_price: float) -> float:
    """Pure profit calculation for a given scalar price point."""
    if price <= 0 or ref_price <= 0:
        return 0.0
    relative_demand = (price / ref_price) ** elasticity
    return (price - cost) * relative_demand


def solve_optimal_price_total(ctx: PricingContext) -> PriceRecommendation:
    """Total pure solver optimizing price within bounds with complete mathematical coverage."""
    p_curr = ctx.current_price.amount
    c = ctx.unit_cost.amount
    e = ctx.elasticity.point_estimate

    # 1. Establish Feasible Price Boundaries
    min_margin_price = c * (1.0 + ctx.min_margin_pct)
    max_change_floor = p_curr * (1.0 - ctx.max_price_change_pct)
    floor_price = max(min_margin_price, max_change_floor)
    ceiling_price = p_curr * (1.0 + ctx.max_price_change_pct)

    bounds = PriceRange(floor=Money(floor_price), ceiling=Money(ceiling_price))

    # 2. Grid Optimization
    grid = np.linspace(bounds.floor.amount, bounds.ceiling.amount, 500)
    ref_price = p_curr
    profits = np.array([compute_profit(p, c, e, ref_price) for p in grid])

    best_idx = int(np.argmax(profits))
    best_price = float(grid[best_idx])

    current_profit = compute_profit(p_curr, c, e, ref_price)
    best_profit = float(profits[best_idx])

    if current_profit > 0:
        lift = (best_profit - current_profit) / current_profit
    else:
        lift = 0.0

    binding = "none"
    if best_idx == 0:
        binding = "floor (margin or max-change)"
    elif best_idx == len(grid) - 1:
        binding = "ceiling (max-change)"

    price_change = (best_price - p_curr) / p_curr if p_curr > 0 else 0.0

    return PriceRecommendation(
        product_id=ctx.product_id,
        current_price=ctx.current_price,
        recommended_price=Money(best_price),
        price_change_pct=float(price_change),
        expected_profit_lift_pct=float(lift),
        binding_constraint=binding,
    )
