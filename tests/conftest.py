"""Fixtures: synthetic sales with known elasticities."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from make_sales import generate


@pytest.fixture(scope="session")
def market():
    return generate(n_products=12, n_weeks=104, seed=7)
