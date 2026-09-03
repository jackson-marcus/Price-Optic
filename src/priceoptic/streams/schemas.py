"""Contract for one weekly sales observation entering the live feed.

The batch pipeline learns elasticities from ``sales.parquet``. The live feed
lets new weeks arrive one observation at a time, so the contract is the row
shape of that file: one (product, week) with the price charged and the units
sold. Anything else is rejected before it can touch an estimate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

LEDGER_NAME = "sales.weekly"


class ObservationError(ValueError):
    """Raised for a payload that cannot be folded into an elasticity estimate."""


@dataclass(frozen=True, slots=True)
class SalesObservation:
    product_id: int
    week: int
    price: float
    units: int

    @property
    def key(self) -> tuple[int, int]:
        """A product sells once per week; this is the idempotency key."""
        return (self.product_id, self.week)


def parse_observation(payload: dict[str, Any]) -> SalesObservation:
    """Validate a raw payload. Prices must be positive and finite; units a
    non-negative whole number (a week with zero sales is a real observation,
    it just cannot enter a log-log fit)."""
    missing = [f for f in ("product_id", "week", "price", "units") if f not in payload]
    if missing:
        raise ObservationError(f"missing fields: {missing}")
    try:
        product_id = int(payload["product_id"])
        week = int(payload["week"])
        price = float(payload["price"])
        units_raw = payload["units"]
        units_f = float(units_raw)
    except (TypeError, ValueError) as exc:
        raise ObservationError(f"non-numeric field: {exc}") from exc
    if isinstance(units_raw, bool) or not math.isfinite(units_f) or units_f != int(units_f):
        raise ObservationError(f"units must be a whole number, got {units_raw!r}")
    units = int(units_f)
    if product_id < 1:
        raise ObservationError(f"product_id must be >= 1, got {product_id}")
    if week < 0:
        raise ObservationError(f"week must be >= 0, got {week}")
    if not math.isfinite(price) or price <= 0:
        raise ObservationError(f"price must be a positive finite number, got {price!r}")
    if units < 0:
        raise ObservationError(f"units must be >= 0, got {units}")
    return SalesObservation(product_id=product_id, week=week, price=price, units=units)
