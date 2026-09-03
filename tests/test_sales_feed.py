"""Live sales feed: contract, exactness against the batch fit, idempotency,
the se gate over time, and what /ingest reports."""

from __future__ import annotations

import math

import pytest

from priceoptic.models.elasticity import estimate_product
from priceoptic.streams.consumer import LedgerConsumer
from priceoptic.streams.producer import SalesLedger
from priceoptic.streams.schemas import ObservationError, SalesObservation, parse_observation
from priceoptic.workers.processor import (
    MIN_WEEKS,
    ElasticityUpdater,
    Fold,
    RunningOLS,
    review_delta,
)


@pytest.mark.parametrize(
    ("payload", "fragment"),
    [
        ({"product_id": 1, "week": 3, "price": 10.0}, "missing"),
        ({"product_id": 1, "week": 3, "price": 10.0, "units": -1}, ">= 0"),
        ({"product_id": 1, "week": 3, "price": 10.0, "units": 2.5}, "whole number"),
        ({"product_id": 1, "week": 3, "price": 0, "units": 5}, "positive"),
        ({"product_id": 1, "week": 3, "price": float("nan"), "units": 5}, "positive"),
        ({"product_id": 0, "week": 3, "price": 10.0, "units": 5}, "product_id"),
        ({"product_id": 1, "week": -1, "price": 10.0, "units": 5}, "week"),
        ({"product_id": "x", "week": 3, "price": 10.0, "units": 5}, "non-numeric"),
    ],
)
def test_contract_rejects_rows_that_cannot_enter_a_fit(payload, fragment):
    with pytest.raises(ObservationError, match=fragment):
        parse_observation(payload)


def test_contract_coerces_numeric_strings():
    obs = parse_observation({"product_id": "4", "week": "10", "price": "19.99", "units": "7.0"})
    assert obs == SalesObservation(product_id=4, week=10, price=19.99, units=7)


def _replayed(market):
    products, sales = market
    ledger = SalesLedger()
    ledger.publish_history(sales)
    consumer = LedgerConsumer(ledger, ElasticityUpdater())
    report = consumer.drain()
    assert report.folded + report.zero_sales == len(sales)
    assert report.duplicates == 0
    return products, sales, consumer


def test_running_fit_matches_batch_ols_after_replay(market):
    products, sales, consumer = _replayed(market)
    compared = 0
    for pid in products["product_id"]:
        live = consumer.updater.estimate(int(pid))
        batch = estimate_product(sales[sales["product_id"] == pid])
        if batch["elasticity"] is None:
            assert live.elasticity is None
            continue
        compared += 1
        # batch rounds to 4 dp, so agreement must be within that rounding
        assert live.elasticity == pytest.approx(batch["elasticity"], abs=1e-4)
        assert live.se == pytest.approx(batch["se"], abs=1e-4)
        assert live.n_obs == batch["n_obs"]
    assert compared >= 10


def test_ledger_reader_resumes_from_its_offset(market):
    _, sales, consumer = _replayed(market)
    offset = consumer.offset
    assert offset == len(sales)
    consumer.ledger.publish([{"product_id": 1, "week": 500, "price": 20.0, "units": 30}])
    report = consumer.drain()
    assert (report.folded, consumer.offset) == (1, offset + 1)
    assert consumer.drain().consumed == 0


def test_redelivery_cannot_shrink_the_standard_error(market):
    """The failure mode the dedup guards against: fold the same weeks twice
    into a raw RunningOLS and the se drops as if you had twice the evidence."""
    products, sales, consumer = _replayed(market)
    pid = int(products["product_id"].iloc[0])
    rows = sales[(sales["product_id"] == pid) & (sales["units"] > 0)]

    naive = RunningOLS()
    for r in rows.itertuples():
        naive.fold(float(r.price), int(r.week), int(r.units))
    once = naive.solve()
    for r in rows.itertuples():
        naive.fold(float(r.price), int(r.week), int(r.units))
    twice = naive.solve()
    assert once is not None and twice is not None
    assert twice[1] < once[1] * 0.75, "double-counting should have inflated confidence"

    before = consumer.updater.estimate(pid)
    consumer.ledger.publish_history(sales[sales["product_id"] == pid])
    report = consumer.drain()
    after = consumer.updater.estimate(pid)
    assert report.duplicates == len(sales[sales["product_id"] == pid])
    assert report.folded == 0
    assert (after.se, after.n_obs) == (before.se, before.n_obs)


def test_zero_sales_week_is_recorded_but_not_fitted():
    updater = ElasticityUpdater()
    obs = SalesObservation(product_id=9, week=0, price=10.0, units=0)
    assert updater.absorb(obs) is Fold.ZERO_SALES
    assert updater.absorb(obs) is Fold.DUPLICATE
    assert updater.estimate(9).n_obs == 0
    assert updater.weeks_seen(9) == 1


def test_product_becomes_priceable_only_after_enough_varied_weeks(market):
    """Feed one product week by week: nothing before MIN_WEEKS, an estimate
    afterwards, and se only ever reported alongside an elasticity."""
    products, sales = market
    pid = int(products["product_id"].iloc[0])
    updater = ElasticityUpdater()
    first_estimate = None
    for r in sales[sales["product_id"] == pid].sort_values("week").itertuples():
        updater.absorb(SalesObservation(pid, int(r.week), float(r.price), int(r.units)))
        est = updater.estimate(pid)
        if est.elasticity is not None and first_estimate is None:
            first_estimate = est
        if est.n_obs < MIN_WEEKS:
            assert est.elasticity is None
            assert not est.priceable
        if est.elasticity is not None:
            assert est.se > 0 and est.ci_lo < est.elasticity < est.ci_hi
    assert first_estimate is not None
    assert first_estimate.n_obs >= MIN_WEEKS
    assert first_estimate.se > updater.estimate(pid).se, "more weeks should tighten the interval"


def test_review_delta_names_what_changed():
    est = {"elasticity": -1.2}
    unpriced = {"estimate": est, "recommendation": None}
    cut = {"estimate": est, "recommendation": {"change_pct": -0.1, "binding_constraint": "none"}}
    raise_ = {
        "estimate": est,
        "recommendation": {"change_pct": 0.25, "binding_constraint": "ceiling (max-change)"},
    }
    gained = review_delta(1, unpriced, cut)
    assert (gained.became_priceable, gained.lost_priceability, gained.direction_flipped) == (
        True,
        False,
        False,
    )
    flipped = review_delta(1, cut, raise_)
    assert flipped.direction_flipped and flipped.binding_changed
    assert not flipped.became_priceable
    lost = review_delta(1, raise_, unpriced)
    assert lost.lost_priceability and not lost.direction_flipped
    assert not math.isnan(flipped.after["recommendation"]["change_pct"])
