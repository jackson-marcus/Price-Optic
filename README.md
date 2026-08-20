# PriceOptic — Econometric Pricing Intelligence & Revenue Optimization

<div align="center">

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![MLflow](https://img.shields.io/badge/MLflow-Registry-0194E2.svg?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Tests: Pytest](https://img.shields.io/badge/tests-pytest-blue.svg?logo=pytest&logoColor=white)](https://pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

> **Econometric pricing engine delivering robust price elasticity of demand estimation with confidence intervals, constrained non-linear profit optimization, and A/B revenue risk simulations.**

---

## 📖 Executive Summary & Value Proposition

**`priceoptic`** is a production-grade, end-to-end machine learning system built with strict engineering discipline, reproducible pipelines, and enterprise MLOps best practices. It bridges the gap between theoretical statistical rigor and high-availability operational microservices.

## 🏷️ Core Methodologies & Econometric Modeling

### 1. Log-Log Econometric Elasticity Estimation
- Fits log-log demand models controlling for seasonal covariates and promotional effects:
$$\ln(Q_i) = lpha + \epsilon \ln(P_i) + \gamma X_i + u_i$$
- Computes robust heteroskedasticity-consistent standard errors and bootstrap 95% confidence intervals for price elasticity $\epsilon$.

### 2. Constrained Profit-Maximizing Price Optimizer
- Solves non-linear optimization maximizing total contribution margin:
$$\max_P \Pi(P) = (P - c) \cdot Q(P) \quad 	ext{s.t.} \quad P_{\min} \le P \le P_{\max}$$
- Incorporates inventory constraints, cross-product cannibalization penalties, and minimum gross margin thresholds.

### 3. A/B Pricing Test Revenue Risk Simulator
- Simulates prospective revenue upside and downside distributions under empirical parameter uncertainty to prevent unhedged margin losses.

## 📊 Architecture & Pipeline

```mermaid
flowchart LR
    Sales[Historical Price & Volume Data] --> Reg[Log-Log Elasticity Estimation<br/>Bootstrap 95% CIs]
    Reg --> Opt[Constrained Profit Optimization<br/>P* Maximizing Margin]
    Opt --> Risk[A/B Revenue Risk Simulator<br/>Downside Uncertainty Bounds]
    Risk --> API[FastAPI :8120] --> UI[Streamlit Pricing Studio :8621]
```

## 🛠️ Tech Stack & Engineering Standards
- **Econometrics & ML:** Python 3.12, NumPy, SciPy Optimize, Statsmodels, Pandas
- **Serving & UI:** FastAPI, Streamlit, MLflow
- **Testing:** Pytest verification of elasticity estimates, convexity of profit curves, and boundary constraints


---

## 🚀 Quickstart & Setup Guide

### 1. Prerequisites & Environment Setup
Using **[uv](https://docs.astral.sh/uv/)** for lightning-fast, reproducible dependency resolution:

```bash
# Clone the repository
git clone https://github.com/jackson-marcus/priceoptic.git
cd priceoptic

# Install dependencies and pre-commit hooks
uv sync --group dev
```

### 2. Run Test Suite & Code Quality Checks
```bash
# Run unit & integration tests with coverage
uv run pytest --cov

# Run ruff linter and formatting checks
uv run ruff check .
uv run ruff format --check .
```

### 3. Launch Services Locally
```bash
# Start FastAPI REST API (listening on port :8120)
make api
# Or: uv run uvicorn priceoptic.api.main:app --reload --port 8120

# Start interactive Streamlit dashboard (listening on port :8621)
make ui

# Launch local MLflow Experiment Tracking UI (listening on port :5013)
make mlflow
```

### 4. Run with Docker Compose
```bash
# Spin up the complete microservice stack
docker compose up --build
```

---

## 📂 Repository Layout

```
priceoptic/
├── .github/workflows/ci.yml       # GitHub Actions CI pipeline (lint, test, build)
├── configs/                      # Configuration files and hyperparameters
├── data/                         # Data directory (raw, interim, processed)
├── scripts/                      # Data generators and operational scripts
├── src/priceoptic/               # Core Python package
│   ├── api/                      # FastAPI routes, schemas, and endpoints
│   ├── models/                   # Statistical models, ML algorithms, and estimators
│   ├── ui/                       # Streamlit interactive application
│   └── settings.py               # Centralized configuration & environment loader
├── tests/                        # Comprehensive Pytest suite
├── docker-compose.yml            # Multi-service container orchestration
├── Dockerfile                    # Container definition for API service
├── Makefile                      # Standardized project tasks
└── pyproject.toml                # Pinned dependencies and tool configs
```

---

## 👤 Author & Contact

**Jackson Marcus**
- **Email:** [jackson.marcus.work@gmail.com](mailto:jackson.marcus.work@gmail.com)
- **Upwork:** [Jackson Marcus on Upwork](https://www.upwork.com/freelancers/~012235717501ad9c7b)
- **GitHub:** [@jackson-marcus](https://github.com/jackson-marcus)

*Available for machine learning engineering, MLOps, data science, and AI system architecture consulting and contract engagements.*

