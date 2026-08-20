"""FastAPI application factory."""

from __future__ import annotations

import logging

from fastapi import FastAPI

from priceoptic import __version__
from priceoptic.api.routes import router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def create_app() -> FastAPI:
    app = FastAPI(
        title="priceoptic",
        description="Price intelligence: per-product elasticity estimation with confidence intervals, constrained profit-maximizing price optimization, and an A/B revenue simulator.",
        version=__version__,
    )
    app.include_router(router)
    return app


app = create_app()
