from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np


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
    pose: str
    quality: dict[str, float | str]
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
    runner_up_score: float | None
    latency_ms: float
    reason: str | None
    bbox: tuple[float, float, float, float] | None
