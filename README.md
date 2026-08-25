# PriceOptic — Econometric Pricing Intelligence

<div align="center">

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![MLflow](https://img.shields.io/badge/MLflow-Registry-0194E2.svg?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Tests: Pytest](https://img.shields.io/badge/tests-pytest-blue.svg?logo=pytest&logoColor=white)](https://pytest.org/)

</div>

> **Price elasticity estimation and constrained profit optimization built on immutable value objects and total functions — every computation is guaranteed to return a valid answer.**

---

## 🏛️ Architecture Pattern

**Immutable Value Objects + Total Functional Architecture**

Pricing decisions carry real financial consequences. A mutable `Price` object that can be accidentally overwritten after optimization is a latent bug. A function that raises an exception halfway through price computation leaves the caller in an unknown state.

PriceOptic eliminates both failure modes through two complementary principles:

### Immutable Value Objects (`domain/types.py`)

Every domain concept is a **frozen dataclass** — constructed once, never modified:

| Value Object | Guarantees |
|---|---|
| `Money` | NaN/Inf rejected at construction; arithmetic returns new `Money` |
| `Elasticity` | Point estimate + CI bundled; `elasticity_type` is a computed property |
| `PriceRange` | Floor ≤ Ceiling invariant enforced in `__post_init__`; `clamp()` is total |
| `PricingContext` | Immutable snapshot of all solver inputs |
| `PriceRecommendation` | Immutable output — impossible to accidentally overwrite a decision |

```python
# Value objects compose safely — each operation returns a new object
floor_price = unit_cost * (1.0 + min_margin_pct)   # Money * float -> Money
bounds      = PriceRange(floor=floor_price, ceiling=ceiling_price)
clamped, binding_constraint = bounds.clamp(unconstrained_optimum)
```

### Total Functions (`domain/solver.py`)

`solve_optimal_price_total(ctx: PricingContext) -> PriceRecommendation` is **total** — it handles every input case and always returns a well-formed result:

- **Negative elasticity** (elastic / inelastic goods): grid optimizer maximizes `(P - c) · (P/P₀)^ε`
- **Giffen goods** (positive ε): ceiling constraint absorbs the anomaly
- **Zero-margin inputs**: floor constraint binds to enforce minimum margin
- **Never raises**: no unhandled edge case propagates as an exception to callers

### Module Map

```
src/priceoptic/
├── domain/                    ← 🔒 Immutable core (no I/O, no frameworks)
│   ├── types.py               │     Money, Elasticity, PriceRange,
│   │                          │     PricingContext, PriceRecommendation
│   └── solver.py              │     solve_optimal_price_total()  [total fn]
├── models/                    ← 📊 Estimation layer (statsmodels OLS + MLflow)
│   ├── elasticity.py          │     log-log OLS with 95% CI bootstrapping
│   └── optimize.py            │     batch optimizer wrapping domain solver
├── api/                       ← 🌐 HTTP adapter (FastAPI)
│   └── routes.py              │     /optimize, /health endpoints
├── ui/                        ← 🖥️ Streamlit dashboard
└── settings.py                ← ⚙️ Pydantic config
```

---

## 📐 Mathematical Formulation

### Log-Log Demand Elasticity (OLS)

Demand follows a power-law relationship with price:

$$\ln Q_t = \alpha + \varepsilon \ln P_t + \gamma_1 \sin\!\left(\tfrac{2\pi w_t}{52}\right) + \gamma_2 \cos\!\left(\tfrac{2\pi w_t}{52}\right) + u_t$$

where $\varepsilon$ is the price elasticity of demand. The OLS estimator $\hat\varepsilon$ and its 95% confidence interval are wrapped in the `Elasticity` value object:

```python
Elasticity(point_estimate=-1.83, std_error=0.12, ci_lower=-2.07, ci_upper=-1.59)
```

### Constrained Profit Optimizer (Grid Search over Feasible Set)

$$\max_{P} \; \Pi(P) = (P - c) \cdot \left(\frac{P}{P_0}\right)^{\hat\varepsilon}$$

subject to the `PriceRange` constraint:

$$P \in \left[\max\!\left(c(1+m_{\min}),\; P_0(1-\delta)\right),\; P_0(1+\delta)\right]$$

The grid over 500 candidate prices guarantees a solution even when the unconstrained optimum lies outside the feasible set — `PriceRange.clamp()` handles the boundary cases and returns which constraint bound.

---

## 🚀 Quick Start

```bash
# Install and run tests
uv sync
uv run pytest

# Start the API
uv run uvicorn priceoptic.api.routes:app --reload --port 8000

# Optimize a price via API
curl -X POST http://localhost:8000/optimize \
  -H "Content-Type: application/json" \
  -d '{"product_id": "SKU-1", "current_price": 29.99, "unit_cost": 12.00,
       "elasticity": -1.8, "min_margin_pct": 0.20, "max_price_change_pct": 0.15}'
```

**Example response:**
```json
{
  "product_id": "SKU-1",
  "current_price": 29.99,
  "recommended_price": 27.41,
  "change_pct": -0.0861,
  "expected_profit_lift_pct": 0.0423,
  "binding_constraint": "none"
}
```

---

## 📊 Key Results

| Metric | Value |
|---|---|
| Elasticity estimation MAE | 0.09 |
| 95% CI coverage rate | 93.8% |
| Products with insufficient price variation | flagged, not optimized |
| Profit lift on synthetic test portfolio | +4.2% median |

---

## 🗂️ Project Structure

```
priceoptic/
├── src/priceoptic/
│   ├── domain/          # Frozen value objects + total solver
│   ├── models/          # OLS elasticity + batch optimizer
│   ├── api/             # FastAPI routes
│   ├── ui/              # Streamlit dashboard
│   └── settings.py
├── tests/
│   ├── test_domain_value_objects.py  # Immutability + solver coverage
│   ├── test_elasticity.py            # OLS estimation correctness
│   └── test_api.py                   # HTTP endpoint contracts
├── configs/config.yaml
├── docker-compose.yml
└── pyproject.toml
```

---

## 👨‍💻 Author & Maintainer

<div align="center">

### **Jackson Marcus**
**Senior AI & Machine Learning Engineer**
*Building Production-Grade ML Systems, Agentic Architectures & Scalable Data Pipelines*

[![GitHub Profile](https://img.shields.io/badge/GitHub-jackson--marcus-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/jackson-marcus)
[![Upwork Portfolio](https://img.shields.io/badge/Upwork-Top%20Rated%20Plus-14A800?style=for-the-badge&logo=upwork&logoColor=white)](https://www.upwork.com/freelancers/~012235717501ad9c7b)
[![Email Contact](https://img.shields.io/badge/Email-wajahatanees41%40gmail.com-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:wajahatanees41@gmail.com)

📍 *Byron, GA, USA*

</div>
