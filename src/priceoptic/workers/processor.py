"""Running per-product elasticity estimates that update one week at a time.

``models/elasticity.py`` refits log(Q) = a + e*log(P) + fourier(week) from
scratch over the whole history. Here the same OLS problem is kept as
sufficient statistics (X'X, X'y, y'y) per product, so a new week folds in
with a 4x4 solve instead of a re-read of the parquet. The estimate is exact:
after replaying the same rows it matches the batch fit, and the tests hold
it to that.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass

import numpy as np

from priceoptic.streams.schemas import SalesObservation

MIN_WEEKS = 20
MIN_DISTINCT_PRICES = 4
N_FEATURES = 4


class Fold(enum.Enum):
    FOLDED = "folded"
    DUPLICATE = "duplicate"
    ZERO_SALES = "zero_sales"


@dataclass(frozen=True, slots=True)
class LiveEstimate:
    product_id: int
    elasticity: float | None
    se: float | None
    ci_lo: float | None
    ci_hi: float | None
    n_obs: int
    n_prices: int
    note: str

    @property
    def priceable(self) -> bool:
        return self.elasticity is not None and self.se is not None and self.se <= 0.5

    def as_dict(self) -> dict:
        return {
            "product_id": self.product_id,
            "elasticity": None if self.elasticity is None else round(self.elasticity, 4),
            "se": None if self.se is None else round(self.se, 4),
            "ci_lo": None if self.ci_lo is None else round(self.ci_lo, 4),
            "ci_hi": None if self.ci_hi is None else round(self.ci_hi, 4),
            "n_obs": self.n_obs,
            "n_prices": self.n_prices,
            "priceable": self.priceable,
            "note": self.note,
        }


def _features(price: float, week: int) -> np.ndarray:
    return np.array(
        [1.0, math.log(price), math.sin(2 * math.pi * week / 52), math.cos(2 * math.pi * week / 52)]
    )


class RunningOLS:
    """Sufficient statistics for one product's log-log demand regression."""

    def __init__(self) -> None:
        self.xtx = np.zeros((N_FEATURES, N_FEATURES))
        self.xty = np.zeros(N_FEATURES)
        self.yty = 0.0
        self.n = 0
        self.prices: set[float] = set()

    def fold(self, price: float, week: int, units: int) -> None:
        x = _features(price, week)
        y = math.log(units)
        self.xtx += np.outer(x, x)
        self.xty += x * y
        self.yty += y * y
        self.n += 1
        self.prices.add(price)

    def solve(self) -> tuple[float, float] | None:
        if self.n < MIN_WEEKS or len(self.prices) < MIN_DISTINCT_PRICES:
            return None
        try:
            beta = np.linalg.solve(self.xtx, self.xty)
            inv = np.linalg.inv(self.xtx)
        except np.linalg.LinAlgError:
            return None
        dof = max(self.n - N_FEATURES, 1)
        # Residual SS at the OLS solution is y'y - b'X'y; clamp the rounding noise.
        sigma2 = max(self.yty - float(beta @ self.xty), 0.0) / dof
        return float(beta[1]), float(math.sqrt(sigma2 * inv[1, 1]))


class ElasticityUpdater:
    """Folds observations into per-product running fits, once per (product, week)."""

    def __init__(self) -> None:
        self._fits: dict[int, RunningOLS] = {}
        self._seen: set[tuple[int, int]] = set()

    def absorb(self, obs: SalesObservation) -> Fold:
        if obs.key in self._seen:
            return Fold.DUPLICATE
        self._seen.add(obs.key)
        fit = self._fits.setdefault(obs.product_id, RunningOLS())
        if obs.units <= 0:
            return Fold.ZERO_SALES
        fit.fold(obs.price, obs.week, obs.units)
        return Fold.FOLDED

    def weeks_seen(self, product_id: int) -> int:
        return sum(1 for pid, _ in self._seen if pid == product_id)

    def estimate(self, product_id: int) -> LiveEstimate:
        fit = self._fits.get(product_id)
        n = fit.n if fit else 0
        n_prices = len(fit.prices) if fit else 0
        solved = fit.solve() if fit else None
        if solved is None:
            return LiveEstimate(
                product_id, None, None, None, None, n, n_prices, "insufficient price variation"
            )
        e, se = solved
        return LiveEstimate(product_id, e, se, e - 1.96 * se, e + 1.96 * se, n, n_prices, "")

    def product_ids(self) -> list[int]:
        return sorted(self._fits)


@dataclass(frozen=True, slots=True)
class ReviewDelta:
    """What one product's recommendation looked like before and after new sales."""

    product_id: int
    before: dict
    after: dict
    became_priceable: bool
    lost_priceability: bool
    direction_flipped: bool
    binding_changed: bool

    def as_dict(self) -> dict:
        return {
            "product_id": self.product_id,
            "before": self.before,
            "after": self.after,
            "became_priceable": self.became_priceable,
            "lost_priceability": self.lost_priceability,
            "direction_flipped": self.direction_flipped,
            "binding_changed": self.binding_changed,
        }


def _direction(rec: dict | None) -> str | None:
    if rec is None:
        return None
    change = rec["change_pct"]
    return "raise" if change > 0 else "cut" if change < 0 else "hold"


def review_delta(product_id: int, before: dict, after: dict) -> ReviewDelta:
    """``before``/``after`` are ``{"estimate": LiveEstimate.as_dict(), "recommendation": rec|None}``."""
    b_rec, a_rec = before["recommendation"], after["recommendation"]
    b_dir, a_dir = _direction(b_rec), _direction(a_rec)
    return ReviewDelta(
        product_id=product_id,
        before=before,
        after=after,
        became_priceable=b_rec is None and a_rec is not None,
        lost_priceability=b_rec is not None and a_rec is None,
        direction_flipped=b_dir is not None and a_dir is not None and b_dir != a_dir,
        binding_changed=b_rec is not None
        and a_rec is not None
        and b_rec["binding_constraint"] != a_rec["binding_constraint"],
    )
