"""Cursor reader that folds ledger entries into an ElasticityUpdater.

The reader owns two things the updater must not care about: where it is in
the ledger, and what happened to each entry (folded, duplicate, or a zero-
sales week that is recorded but cannot enter a log-log fit).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from priceoptic.streams.producer import SalesLedger
from priceoptic.workers.processor import ElasticityUpdater, Fold


@dataclass
class DrainReport:
    folded: int = 0
    duplicates: int = 0
    zero_sales: int = 0
    touched: set[int] = field(default_factory=set)

    @property
    def consumed(self) -> int:
        return self.folded + self.duplicates + self.zero_sales


class LedgerConsumer:
    def __init__(self, ledger: SalesLedger, updater: ElasticityUpdater) -> None:
        self.ledger = ledger
        self.updater = updater
        self.offset = 0

    def drain(self) -> DrainReport:
        """Fold every unread entry. Re-delivery is safe: the updater refuses a
        (product, week) it has already seen, so draining the same entries
        twice cannot shrink a standard error."""
        report = DrainReport()
        for offset, obs in self.ledger.read_from(self.offset):
            outcome = self.updater.absorb(obs)
            if outcome is Fold.FOLDED:
                report.folded += 1
                report.touched.add(obs.product_id)
            elif outcome is Fold.DUPLICATE:
                report.duplicates += 1
            else:
                report.zero_sales += 1
                report.touched.add(obs.product_id)
            self.offset = offset + 1
        return report
