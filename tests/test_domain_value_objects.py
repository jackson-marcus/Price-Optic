"""Unit tests for Immutable Value Objects and Total Functions in PriceOptic."""

import dataclasses

import pytest

from priceoptic.domain.solver import solve_optimal_price_total
from priceoptic.domain.types import (
    Elasticity,
    ElasticityType,
    Money,
    PriceRange,
    PricingContext,
)


def test_money_immutability_and_arithmetic():
    m1 = Money(10.50)
    m2 = Money(4.25)

    m3 = m1 + m2
    assert m3.amount == 14.75

    m4 = m1 * 2.0
    assert m4.amount == 21.0

    with pytest.raises(dataclasses.FrozenInstanceError):
        m1.amount = 99.0  # Frozen dataclass should reject direct mutation


def test_elasticity_classification():
    e_elastic = Elasticity(point_estimate=-2.5)
    assert e_elastic.elasticity_type == ElasticityType.ELASTIC

    e_inelastic = Elasticity(point_estimate=-0.4)
    assert e_inelastic.elasticity_type == ElasticityType.INELASTIC


def test_price_range_clamping():
    bounds = PriceRange(floor=Money(8.0), ceiling=Money(12.0))

    clamped_low, reason_low = bounds.clamp(Money(5.0))
    assert clamped_low.amount == 8.0
    assert reason_low == "floor"

    clamped_hi, reason_hi = bounds.clamp(Money(20.0))
    assert clamped_hi.amount == 12.0
    assert reason_hi == "ceiling"

    clamped_mid, reason_mid = bounds.clamp(Money(10.0))
    assert clamped_mid.amount == 10.0
    assert reason_mid == "none"


def test_total_solver_execution():
    ctx = PricingContext(
        product_id="SKU-1",
        current_price=Money(10.0),
        unit_cost=Money(4.0),
        elasticity=Elasticity(point_estimate=-1.8),
        min_margin_pct=0.20,
        max_price_change_pct=0.15,
    )

    rec = solve_optimal_price_total(ctx)

    assert rec.product_id == "SKU-1"
    assert rec.recommended_price.amount >= 8.5
    assert rec.recommended_price.amount <= 11.5
    assert rec.expected_profit_lift_pct >= 0.0
