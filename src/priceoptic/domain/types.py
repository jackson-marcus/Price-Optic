"""Immutable Value Objects & Total Domain Types.

Frozen, self-validating value objects and total types representing economic pricing constructs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class ElasticityType(StrEnum):
    ELASTIC = "elastic"  # e < -1
    INELASTIC = "inelastic"  # -1 <= e <= 0
    GIFFEN = "giffen"  # e > 0
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Money:
    """Immutable monetary amount with safe arithmetic."""

    amount: float

    def __post_init__(self) -> None:
        if math.isnan(self.amount) or math.isinf(self.amount):
            raise ValueError("Money amount cannot be NaN or Inf")
        object.__setattr__(self, "amount", round(float(self.amount), 4))

    def __add__(self, other: Money) -> Money:
        return Money(self.amount + other.amount)

    def __sub__(self, other: Money) -> Money:
        return Money(self.amount - other.amount)

    def __mul__(self, factor: float) -> Money:
        return Money(self.amount * factor)

    def __truediv__(self, factor: float) -> Money:
        if factor == 0:
            raise ZeroDivisionError("Cannot divide Money by zero")
        return Money(self.amount / factor)

    def as_float(self) -> float:
        return self.amount


@dataclass(frozen=True)
class Elasticity:
    """Price elasticity of demand with confidence intervals and classification."""

    point_estimate: float
    std_error: float = 0.0
    ci_lower: float = 0.0
    ci_upper: float = 0.0

    @property
    def elasticity_type(self) -> ElasticityType:
        if self.point_estimate < -1.0:
            return ElasticityType.ELASTIC
        if -1.0 <= self.point_estimate <= 0.0:
            return ElasticityType.INELASTIC
        if self.point_estimate > 0.0:
            return ElasticityType.GIFFEN
        return ElasticityType.UNKNOWN

    @property
    def is_statistically_significant(self) -> bool:
        return not (self.ci_lower <= 0.0 <= self.ci_upper)


@dataclass(frozen=True)
class PriceRange:
    """Clamped feasible price boundary [min_price, max_price]."""

    floor: Money
    ceiling: Money

    def __post_init__(self) -> None:
        if self.floor.amount > self.ceiling.amount:
            # Clamp floor to ceiling to maintain valid total interval
            object.__setattr__(self, "floor", self.ceiling)

    def clamp(self, target: Money) -> tuple[Money, str]:
        """Clamps candidate price into the valid range and returns binding constraint label."""
        if target.amount < self.floor.amount:
            return self.floor, "floor"
        if target.amount > self.ceiling.amount:
            return self.ceiling, "ceiling"
        return target, "none"


@dataclass(frozen=True)
class PricingContext:
    """Immutable problem context for price optimization."""

    product_id: str
    current_price: Money
    unit_cost: Money
    elasticity: Elasticity
    min_margin_pct: float = 0.15
    max_price_change_pct: float = 0.20


@dataclass(frozen=True)
class PriceRecommendation:
    """Total output value object representing a pricing decision."""

    product_id: str
    current_price: Money
    recommended_price: Money
    price_change_pct: float
    expected_profit_lift_pct: float
    binding_constraint: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "product_id": self.product_id,
            "current_price": round(self.current_price.amount, 2),
            "recommended_price": round(self.recommended_price.amount, 2),
            "change_pct": round(self.price_change_pct, 4),
            "expected_profit_lift_pct": round(self.expected_profit_lift_pct, 4),
            "binding_constraint": self.binding_constraint,
        }
