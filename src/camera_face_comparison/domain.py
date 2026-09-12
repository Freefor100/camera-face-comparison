from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from .open_set_policy import AcceptanceRule


@dataclass(frozen=True)
class Person:
    """本地标准人脸库中的一个命名身份。"""

    id: str
    display_name: str
    created_at: datetime


@dataclass(frozen=True)
class FaceSample:
    """一张已录入的人脸图片及其特征向量。"""

    id: str
    person_id: str
    image_path: str
    embedding: np.ndarray
    quality_metrics: dict[str, float]
    created_at: datetime
    source_type: str = "camera"
    image_sha256: str | None = None
    embedding_sha256: str | None = None


@dataclass(frozen=True)
class RecognitionResult:
    """一次识别尝试的、可直接交给界面展示的结果。"""

    status: str
    person_id: str | None
    display_name: str | None
    top_score: float | None
    second_score: float | None
    score_gap: float | None
    acceptance_score: float | None
    acceptance_rule: AcceptanceRule
    latency_ms: float
    reason: str | None
    bbox: tuple[float, float, float, float] | None
    quality_metrics: dict[str, float]
    quality_warnings: tuple[str, ...]
    frame_count: int = 1
    valid_frame_count: int = 1
    selected_frame_index: int | None = None
    selected_method: str | None = None
