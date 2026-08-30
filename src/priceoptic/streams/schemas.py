"""Event contracts for the price.ticks stream."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

EVENT_NAME = "PriceTick"
STREAM_NAME = "price.ticks"
REQUIRED_FIELDS: tuple[str, ...] = ("sku", "price", "competitor")


@dataclass(frozen=True)
class StreamEvent:
    event_id: str
    event_type: str
    payload: dict[str, Any]
    headers: dict[str, str] = field(default_factory=dict)

    @classmethod
    def create(cls, payload: dict[str, Any], **headers: str) -> StreamEvent:
        missing = [name for name in REQUIRED_FIELDS if name not in payload]
        if missing:
            raise ValueError(f"missing fields: {missing}")
        return cls(
            event_id=headers.pop("event_id", uuid4().hex),
            event_type=EVENT_NAME,
            payload=dict(payload),
            headers=dict(headers),
        )
