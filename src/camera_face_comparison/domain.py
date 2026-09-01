from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np


@dataclass(frozen=True)
class Person:
    """A named identity in the local face library."""

    id: str
    display_name: str
    created_at: datetime


@dataclass(frozen=True)
class FaceSample:
    """One enrolled face image and its extracted embedding."""

    id: str
    person_id: str
    image_path: str
    embedding: np.ndarray
    pose: str
    quality: dict[str, float | str]
    created_at: datetime
    source_type: str = "camera"
    image_sha256: str | None = None
    embedding_sha256: str | None = None


@dataclass(frozen=True)
class RecognitionResult:
    """The UI-ready outcome of a single recognition attempt."""

    status: str
    person_id: str | None
    display_name: str | None
    top_score: float | None
    runner_up_score: float | None
    latency_ms: float
    reason: str | None
    bbox: tuple[float, float, float, float] | None
