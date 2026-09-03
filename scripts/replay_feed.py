"""Replay the sales history week by week through the live feed and measure
what the se gate buys you.

Every week the running estimate for each product is re-solved and re-priced.
A recommendation is "wrong-way" when it says raise while the true-elasticity
optimum says cut (or vice versa), and a "reversal" when it points the other
way from the previous week it was allowed to price. The gate refuses to price
while se > threshold; a stricter gate waits longer but should flip less.

Usage:
    uv run python scripts/replay_feed.py            # uses data/processed/*.parquet
    uv run python scripts/replay_feed.py --weekly   # also print week-by-week priceable counts
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from priceoptic.models.optimize import optimize_price
from priceoptic.settings import get_config, resolve_path
from priceoptic.streams.consumer import LedgerConsumer
from priceoptic.streams.producer import SalesLedger
from priceoptic.workers.processor import ElasticityUpdater

GATES = (np.inf, 1.0, 0.5, 0.3, 0.2)


def direction(change_pct: float) -> str:
    return "raise" if change_pct > 0 else "cut" if change_pct < 0 else "hold"


def replay(products: pd.DataFrame, sales: pd.DataFrame) -> pd.DataFrame:
    """One row per (product, week) with the live estimate and its recommendation."""
    ledger = SalesLedger()
    consumer = LedgerConsumer(ledger, ElasticityUpdater())
    meta = products.set_index("product_id")
    rows = []
    for week, chunk in sales.sort_values("week").groupby("week", sort=True):
        ledger.publish_history(chunk)
        consumer.drain()
        for pid in meta.index:
            est = consumer.updater.estimate(int(pid))
            row = {"product_id": int(pid), "week": int(week), "se": est.se, "n_obs": est.n_obs}
            if est.elasticity is not None:
                rec = optimize_price(
                    float(meta.at[pid, "base_price"]), float(meta.at[pid, "cost"]), est.elasticity
                )
                row["direction"] = direction(rec.change_pct)
                row["recommended_price"] = rec.recommended_price
            rows.append(row)
    frame = pd.DataFrame(rows)
    truth = {
        int(pid): direction(
            optimize_price(
                float(r["base_price"]), float(r["cost"]), float(r["true_elasticity"])
            ).change_pct
        )
        for pid, r in meta.iterrows()
    }
    frame["true_direction"] = frame["product_id"].map(truth)
    return frame


def score(frame: pd.DataFrame, gate: float) -> dict:
    allowed = frame[frame["se"].notna() & (frame["se"] <= gate)].sort_values(["product_id", "week"])
    first = allowed.groupby("product_id")["week"].min()
    n_weeks = int(frame["week"].max()) + 1
    reversals = 0
    for _, g in allowed.groupby("product_id"):
        d = g["direction"].to_numpy()
        reversals += int((d[1:] != d[:-1]).sum())
    wrong = int((allowed["direction"] != allowed["true_direction"]).sum())
    return {
        "gate": "none" if np.isinf(gate) else f"{gate:.1f}",
        "products_priced": int(first.size),
        "median_first_week": float(first.median()) if first.size else float("nan"),
        "priceable_product_weeks": len(allowed),
        "reversals": reversals,
        "wrong_way_weeks": wrong,
        "wrong_way_pct": 100.0 * wrong / len(allowed) if len(allowed) else float("nan"),
        "weeks_forgone_pct": 100.0 * (1 - len(allowed) / (n_weeks * frame["product_id"].nunique())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weekly", action="store_true")
    args = parser.parse_args()
    processed = resolve_path(get_config()["data"]["processed_dir"])
    products = pd.read_parquet(processed / "products.parquet")
    sales = pd.read_parquet(processed / "sales.parquet")
    frame = replay(products, sales)
    table = pd.DataFrame([score(frame, g) for g in GATES])
    pd.set_option("display.width", 160)
    print(f"{products.shape[0]} products x {frame['week'].max() + 1} weeks replayed\n")
    print(table.round(1).to_string(index=False))
    if args.weekly:
        weekly = (
            frame[frame["se"].notna()].groupby("week")["se"].apply(lambda s: int((s <= 0.5).sum()))
        )
        print("\npriceable products per week at se<=0.5:")
        print(weekly.to_string())


if __name__ == "__main__":
    main()
