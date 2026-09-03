"""Append-only ledger of weekly sales observations.

Every observation gets a monotonically increasing offset, so a reader can
resume from wherever it stopped. The ledger never validates economics or
deduplicates - that is the consumer's job - it only refuses payloads that do
not parse, and keeps those as rejects so a caller can see what was dropped.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import pandas as pd

from priceoptic.streams.schemas import (
    LEDGER_NAME,
    ObservationError,
    SalesObservation,
    parse_observation,
)


@dataclass(frozen=True, slots=True)
class Reject:
    payload: dict[str, Any]
    reason: str


class SalesLedger:
    def __init__(self, name: str = LEDGER_NAME) -> None:
        self.name = name
        self._entries: list[SalesObservation] = []

    def __len__(self) -> int:
        return len(self._entries)

    def append(self, obs: SalesObservation) -> int:
        self._entries.append(obs)
        return len(self._entries) - 1

    def read_from(self, offset: int) -> list[tuple[int, SalesObservation]]:
        """Entries with offset >= ``offset`` in arrival order."""
        return list(enumerate(self._entries))[offset:]

    def publish(self, payloads: Iterable[dict[str, Any]]) -> tuple[list[int], list[Reject]]:
        """Parse and append a batch. Malformed payloads are returned, not raised,
        so one bad row does not block the rest of the batch."""
        offsets: list[int] = []
        rejects: list[Reject] = []
        for payload in payloads:
            try:
                obs = parse_observation(payload)
            except ObservationError as exc:
                rejects.append(Reject(payload=dict(payload), reason=str(exc)))
                continue
            offsets.append(self.append(obs))
        return offsets, rejects

    def publish_history(self, sales: pd.DataFrame) -> int:
        """Replay a ``sales.parquet`` frame week by week, in week order across
        products - the order a real feed would deliver it. Returns rows added."""
        ordered = sales.sort_values(["week", "product_id"], kind="stable")
        before = len(self)
        for row in ordered.itertuples(index=False):
            self.append(
                SalesObservation(
                    product_id=int(row.product_id),
                    week=int(row.week),
                    price=float(row.price),
                    units=int(row.units),
                )
            )
        return len(self) - before
