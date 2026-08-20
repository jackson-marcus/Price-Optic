"""Elasticity estimation: log-log OLS per product with seasonality controls.

log(Q) = a + e*log(P) + fourier(week) — e is the price elasticity. Standard
errors give a CI; products whose CI is too wide are flagged instead of
optimized (bad estimates make bad prices).
"""

from __future__ import annotations

import logging
import pickle

import mlflow
import numpy as np
import pandas as pd

from priceoptic.settings import get_config, get_settings, resolve_path

logger = logging.getLogger(__name__)


def _design(price: np.ndarray, week: np.ndarray) -> np.ndarray:
    """[1, log(P), sin, cos] design matrix (annual seasonality control)."""
    return np.column_stack(
        [
            np.ones(len(price)),
            np.log(price),
            np.sin(2 * np.pi * week / 52),
            np.cos(2 * np.pi * week / 52),
        ]
    )


def estimate_product(sales: pd.DataFrame) -> dict:
    df = sales[sales["units"] > 0]
    if len(df) < 20 or df["price"].nunique() < 4:
        return {"elasticity": None, "se": None, "note": "insufficient price variation"}
    x = _design(df["price"].to_numpy(), df["week"].to_numpy())
    y = np.log(df["units"].to_numpy())
    beta, residuals, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    dof = max(len(y) - x.shape[1], 1)
    sigma2 = (
        float(residuals[0]) / dof if len(residuals) else float(((y - x @ beta) ** 2).sum()) / dof
    )
    cov = sigma2 * np.linalg.inv(x.T @ x)
    se = float(np.sqrt(cov[1, 1]))
    return {
        "elasticity": round(float(beta[1]), 4),
        "se": round(se, 4),
        "ci_lo": round(float(beta[1] - 1.96 * se), 4),
        "ci_hi": round(float(beta[1] + 1.96 * se), 4),
        "n_obs": len(df),
        "note": "",
    }


def estimate_all() -> pd.DataFrame:
    cfg = get_config()
    processed = resolve_path(cfg["data"]["processed_dir"])
    products = pd.read_parquet(processed / "products.parquet")
    sales = pd.read_parquet(processed / "sales.parquet")

    mlflow.set_tracking_uri(get_settings().mlflow_tracking_uri)
    mlflow.set_experiment(cfg["eval"]["experiment_name"])

    rows = []
    for _, product in products.iterrows():
        est = estimate_product(sales[sales["product_id"] == product["product_id"]])
        rows.append({"product_id": product["product_id"], **est})
    result = products.merge(pd.DataFrame(rows), on="product_id")

    ok = result[result["elasticity"].notna()]
    mae = float((ok["elasticity"] - ok["true_elasticity"]).abs().mean())
    coverage = float(
        ((ok["true_elasticity"] >= ok["ci_lo"]) & (ok["true_elasticity"] <= ok["ci_hi"])).mean()
    )
    with mlflow.start_run(run_name="elasticity-ols"):
        mlflow.log_params({"n_products": len(products)})
        mlflow.log_metrics({"elasticity_mae": mae, "ci_coverage": coverage, "n_estimated": len(ok)})
    logger.info(
        "elasticity MAE %.3f | 95%% CI coverage %.2f | estimated %d/%d",
        mae,
        coverage,
        len(ok),
        len(products),
    )

    artifacts = resolve_path(cfg["data"]["artifacts_dir"])
    artifacts.mkdir(parents=True, exist_ok=True)
    result.to_parquet(artifacts / "elasticities.parquet", index=False)
    with open(artifacts / "meta.pkl", "wb") as f:
        pickle.dump({"mae": mae, "coverage": coverage}, f)
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    estimate_all()
