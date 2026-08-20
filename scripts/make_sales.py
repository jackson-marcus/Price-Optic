"""Synthetic weekly sales with known price elasticities.

Each product has a true constant elasticity; historical prices vary through
promos and repricing, demand follows Q = a * P^elasticity * season * noise.
True parameters are stored so estimation quality is measurable.

Usage:
    uv run python scripts/make_sales.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from priceoptic.settings import get_config, resolve_path

CATEGORIES = {
    "staples": (-0.8, -0.4),  # inelastic
    "electronics": (-2.6, -1.4),  # elastic
    "apparel": (-2.0, -1.0),
    "luxury": (-1.2, -0.6),
}


def generate(n_products: int, n_weeks: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    cats = rng.choice(list(CATEGORIES), n_products)
    elasticity = np.array([rng.uniform(*CATEGORIES[c]) for c in cats])
    base_price = rng.uniform(8, 120, n_products).round(2)
    cost = (base_price * rng.uniform(0.4, 0.75, n_products)).round(2)
    base_demand = rng.uniform(80, 900, n_products)

    rows = []
    for p in range(n_products):
        # Price history: occasional promos (-10..30%) and small repricings.
        price = base_price[p]
        for week in range(n_weeks):
            if rng.random() < 0.18:
                price = base_price[p] * rng.uniform(0.7, 1.1)
            season = (
                1.0 + 0.25 * np.sin(2 * np.pi * week / 52) + (0.5 if week % 52 in (46, 47) else 0.0)
            )
            q = base_demand[p] * (price / base_price[p]) ** elasticity[p] * season
            q *= rng.lognormal(0, 0.18)  # demand noise caps estimation quality
            rows.append(
                {
                    "product_id": p + 1,
                    "week": week,
                    "price": round(float(price), 2),
                    "units": max(int(q), 0),
                }
            )
    sales = pd.DataFrame(rows)
    products = pd.DataFrame(
        {
            "product_id": np.arange(1, n_products + 1),
            "category": cats,
            "base_price": base_price,
            "cost": cost,
            "true_elasticity": elasticity.round(4),
        }
    )
    return products, sales


def main() -> None:
    cfg = get_config()["data"]
    products, sales = generate(cfg["n_products"], cfg["n_weeks"], cfg["seed"])
    out = resolve_path(cfg["processed_dir"])
    out.mkdir(parents=True, exist_ok=True)
    products.to_parquet(out / "products.parquet", index=False)
    sales.to_parquet(out / "sales.parquet", index=False)
    print(f"Wrote {len(products)} products x {cfg['n_weeks']} weeks -> {out}")


if __name__ == "__main__":
    main()
