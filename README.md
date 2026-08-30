<div align="center">

<img src="docs/brand/banner.svg" alt="PriceOptic — Econometric Pricing Intelligence" width="720">

</div>

# PriceOptic — Econometric Pricing Intelligence

**Estimate how demand responds to price, then find the price that maximises profit within the margins you can actually live with.** PriceOptic fits a per-product price-elasticity model with confidence intervals, feeds it into a constrained profit optimiser, and simulates what a proposed price change would do to revenue — including the odds it backfires. The domain core is built from immutable value objects and total functions, so a pricing computation can never leave a caller holding a half-finished decision or a silently mutated number.

<div align="center">

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Tests: pytest](https://img.shields.io/badge/tests-pytest-0A9EDC.svg?logo=pytest&logoColor=white)](https://pytest.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)

</div>

---

## The problem

"What price should we set?" hides two hard questions. First, *how sensitive is demand to price?* — the price elasticity, which you have to estimate from noisy historical sales where price and season move at the same time. Second, *given that estimate, what price is best?* — and "best" is never unconstrained: you have a minimum margin to protect and a limit on how far you dare move from today's price in one step.

PriceOptic answers both, and adds a third the naive workflow skips: *should we trust the change?* Elasticity estimates come with error bars, and a point recommendation hides that uncertainty. So before a price ships, PriceOptic simulates it — bootstrapping the elasticity and demand noise — to report the distribution of the profit swing and the probability the change actually helps.

## What it does

- **Elasticity estimation** — fits a log-log demand model per product with seasonality controls and reports a 95% confidence interval. Products whose interval is too wide are flagged, not optimised, because a bad estimate makes a bad price.
- **Constrained price optimisation** — searches for the profit-maximising price inside a feasible band set by your minimum margin and maximum allowed price move.
- **A/B revenue simulation** — Monte-Carlo simulates a candidate price against the current one under estimation uncertainty and reports the profit/revenue difference distribution and the probability of a positive outcome.

## How it works

```mermaid
flowchart LR
    G["Synthetic weekly sales<br/>(scripts/make_sales.py)"] --> E
    subgraph EST["Estimation (models/)"]
        E["elasticity.py<br/>log-log OLS + 95% CI"] --> A["elasticities.parquet<br/>(data/artifacts)"]
    end
    A --> API
    subgraph SRV["FastAPI service (api/)"]
        API["/products · /optimize · /simulate · /health"]
        API --> O["optimize.py<br/>constrained grid optimiser"]
        API --> S["simulator/ab.py<br/>bootstrap A/B simulation"]
    end
    API --> UI["Streamlit dashboard (ui/)"]
```

The pipeline is one-directional: generate sales → estimate elasticities into a parquet artifact → serve optimisation and simulation over that artifact. The Streamlit UI is a thin client that talks to the API over HTTP.

### Domain design: immutable value objects + total functions

Pricing decisions carry financial consequences, so the domain layer (`domain/`) is written to make two classes of bug unrepresentable. It has no I/O and no framework imports.

**Immutable value objects (`domain/types.py`)** — every domain concept is a frozen dataclass, constructed once and never mutated, validating itself on construction:

| Value object | Guarantee |
|---|---|
| `Money` | Rejects NaN/Inf at construction; arithmetic returns a new `Money` |
| `Elasticity` | Bundles point estimate + CI; `elasticity_type` and significance are computed properties |
| `PriceRange` | Restores the floor ≤ ceiling invariant in `__post_init__`; `clamp()` is total |
| `PricingContext` | Immutable snapshot of every solver input |
| `PriceRecommendation` | Immutable output — a decision cannot be accidentally overwritten |

**Total solver (`domain/solver.py`)** — `solve_optimal_price_total(ctx) -> PriceRecommendation` handles every input case and always returns a well-formed result. Elastic, inelastic and Giffen (positive-ε) goods are all covered by the same constrained grid search; the `PriceRange` clamp absorbs any optimum that falls outside the feasible band, and the function never raises on an economic edge case.

The production pipeline uses a parallel, config-driven implementation of the same optimisation in `models/optimize.py`, which reads the margin and price-move limits from `configs/config.yaml`.

## Methodology

### Elasticity: log-log OLS with seasonality controls

Demand is modelled as a constant-elasticity power law in price, with an annual Fourier seasonality control:

$$\ln Q_t = \alpha + \varepsilon \ln P_t + \gamma_1 \sin\!\left(\tfrac{2\pi w_t}{52}\right) + \gamma_2 \cos\!\left(\tfrac{2\pi w_t}{52}\right) + u_t$$

The coefficient $\varepsilon$ is the price elasticity of demand. It is fit by ordinary least squares (`numpy.linalg.lstsq`); the standard error comes from the OLS covariance $\sigma^2 (X^\top X)^{-1}$, and the reported interval is $\hat\varepsilon \pm 1.96\,\mathrm{se}$. Products with fewer than 20 valid weeks or too little price variation are skipped, and the optimiser refuses any product whose standard error exceeds 0.5.

### Constrained profit optimisation

For a candidate price $P$, reference price $P_0$ and unit cost $c$, profit under constant elasticity is:

$$\Pi(P) = (P - c) \cdot \left(\frac{P}{P_0}\right)^{\hat\varepsilon}$$

maximised over the feasible band

$$P \in \left[\max\!\left(c(1+m_{\min}),\; P_0(1-\delta)\right),\; P_0(1+\delta)\right]$$

where $m_{\min}$ is the minimum margin and $\delta$ the maximum price move. The optimiser evaluates 500 grid points across the band, so it returns a valid answer for any demand shape (including inelastic goods that push straight to the ceiling) and reports which constraint, if any, binds.

### A/B revenue simulation

To size the risk of a price change, `simulator/ab.py` draws the elasticity from $\mathcal{N}(\hat\varepsilon, \mathrm{se})$, adds lognormal weekly demand noise to both the current and candidate arms, and reports the mean, 5th and 95th percentiles of the profit difference plus the probability it is positive.

## Getting started

Requires [uv](https://github.com/astral-sh/uv) and Python 3.12.

```bash
make install                                 # uv sync --group dev

uv run python scripts/make_sales.py          # generate synthetic sales -> data/processed
uv run python -m priceoptic.models.elasticity  # estimate elasticities -> data/artifacts

make api                                     # FastAPI on http://localhost:8120
make ui                                      # Streamlit dashboard on http://localhost:8621
```

Run the estimation step before starting the API — the endpoints read the `elasticities.parquet` artifact and return `503` until it exists.

Or with Docker:

```bash
make docker-up                               # api on :8120, ui on :8621
make docker-down
```

## API

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `GET` | `/products` | Estimated vs true elasticity per product, with 95% CIs |
| `GET` | `/optimize/{product_id}` | Constrained profit-maximising price for a product |
| `POST` | `/simulate` | A/B simulate a candidate price (`{"product_id": 1, "candidate_price": 27.5}`) |

Example — recommend a price for product 1:

```bash
curl http://localhost:8120/optimize/1
```

```jsonc
// Illustrative shape only — values depend on the synthetic dataset and seed.
{
  "product_id": 1,
  "elasticity": -1.83,
  "current_price": 29.99,
  "recommended_price": 27.41,
  "change_pct": -0.0861,
  "expected_profit_lift_pct": 0.0423,
  "binding_constraint": "none"
}
```

`/optimize` returns `422` when a product's elasticity is not estimable or its confidence interval is too wide to price on.

## Evaluation

Because the sales data is synthetic, every product carries a **known true elasticity**, so estimation quality can be measured directly. `models/elasticity.py` reports the mean absolute error between the estimated and true elasticities and the 95%-CI coverage (how often the true value falls inside the estimated interval), and logs both to MLflow. To reproduce:

```bash
uv run python scripts/make_sales.py
uv run python -m priceoptic.models.elasticity    # prints MAE and CI coverage
make mlflow                                       # optional: MLflow UI on http://localhost:5013
```

No fixed numbers are quoted here — they depend on the generated dataset and seed. Run the command to produce them for your configuration.

## Testing

```bash
make test                                    # uv run pytest --cov
```

- `test_domain_value_objects.py` — value-object immutability/validation and total-solver coverage
- `test_elasticity.py` — OLS estimation on synthetic data with known parameters
- `test_api.py` — HTTP endpoint contracts

## Limitations

- Estimation accuracy is capped by price variation in the history — products that rarely reprice have wide intervals and are refused rather than priced.
- The demand model assumes constant elasticity and a single annual seasonality term; real demand has cross-price effects, competitor moves and trends this ignores.
- The optimiser assumes the estimated elasticity holds at the recommended price; large moves extrapolate beyond the observed price range.
- All bundled data is synthetic; thresholds and priors would need recalibration on real sales.

## Project structure

```
src/priceoptic/
├── domain/         # Immutable value objects + total solver (no I/O, no frameworks)
│   ├── types.py    #   Money, Elasticity, PriceRange, PricingContext, PriceRecommendation
│   └── solver.py   #   solve_optimal_price_total() — pure, total
├── models/         # elasticity.py (log-log OLS + CI), optimize.py (constrained grid optimiser)
├── simulator/      # ab.py — bootstrap A/B revenue/profit simulation
├── api/            # main.py (app factory) + routes.py (endpoints)
├── ui/             # app.py — Streamlit dashboard
└── settings.py     # pydantic-settings + configs/config.yaml loader
scripts/make_sales.py   # synthetic sales generator with known true elasticities
```

## License

MIT

---

<div align="center">

**Jackson Marcus** · Senior AI & Machine Learning Engineer

[![GitHub](https://img.shields.io/badge/GitHub-jackson--marcus-181717?logo=github&logoColor=white)](https://github.com/jackson-marcus)
[![Email](https://img.shields.io/badge/Email-contact-D14836?logo=gmail&logoColor=white)](mailto:wajahatanees41@gmail.com)

</div>
