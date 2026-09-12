from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np

from .face_engine import normalize_embedding


@dataclass(frozen=True)
class MultiFrameObservation:
    """短时间采集的一帧有效人脸及其质量信息。"""

    frame: np.ndarray
    embedding: np.ndarray
    bbox: tuple[float, float, float, float]
    quality_metrics: dict[str, float]
    quality_warnings: tuple[str, ...]
    frame_index: int


def select_sharpest_frame(
    frames: Sequence[MultiFrameObservation],
) -> MultiFrameObservation:
    """返回人脸区域清晰度最高的一帧。"""

    _require_frames(frames)
    return max(frames, key=lambda item: (_blur_value(item), -item.frame_index))


def select_consistent_frame(
    frames: Sequence[MultiFrameObservation],
) -> MultiFrameObservation:
    """返回与其余有效帧平均余弦相似度最高的一帧。

    相似度相同时优先选择清晰度更高的帧，再以采集顺序稳定打破剩余平局。
    """

    _require_frames(frames)
    normalized = np.stack([normalize_embedding(item.embedding) for item in frames])
    if len(frames) == 1:
        return frames[0]
    similarities = normalized @ normalized.T
    similarity_means = (similarities.sum(axis=1) - 1.0) / (len(frames) - 1)
    best_index = max(
        range(len(frames)),
        key=lambda index: (
            float(similarity_means[index]),
            _blur_value(frames[index]),
            -frames[index].frame_index,
        ),
    )
    return frames[best_index]


def select_mean_embedding(
    frames: Sequence[MultiFrameObservation],
) -> MultiFrameObservation:
    """平均全部有效单位特征，并用最清晰帧作为代表画面。"""

    _require_frames(frames)
    normalized = np.stack([normalize_embedding(item.embedding) for item in frames])
    fused_embedding = normalize_embedding(np.mean(normalized, axis=0))
    representative = select_sharpest_frame(frames)
    return replace(representative, embedding=fused_embedding)


def _require_frames(frames: Sequence[MultiFrameObservation]) -> None:
    """保证选择器至少收到一帧有效人脸观察。"""

    if not frames:
        raise ValueError("no_valid_frames")


def _blur_value(frame: MultiFrameObservation) -> float:
    """读取清晰度并把缺失或非有限值视为最低优先级。"""

    value = float(frame.quality_metrics.get("blur_variance", float("-inf")))
    return value if np.isfinite(value) else float("-inf")
