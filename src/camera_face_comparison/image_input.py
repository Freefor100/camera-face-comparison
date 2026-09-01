from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from .config import Settings

if TYPE_CHECKING:
    from .face_engine import FaceObservation


SourceType = Literal["camera", "file", "dataset"]


@dataclass(frozen=True)
class ImageInput:
    """进入录入、识别或评测流程的一张 BGR 图像。"""

    frame: np.ndarray
    source_type: SourceType
    safe_name: str | None

    @classmethod
    def from_camera(cls, frame: np.ndarray) -> ImageInput:
        """复制摄像头当前帧，避免后台任务读取可变缓冲区。

        参数：
            frame：摄像头产生的 BGR 图像。
        返回：
            来源标记为摄像头的独立图片输入。
        前置条件：
            输入应为非空的三通道 NumPy 图像数组。
        """
        return cls(frame=_validated_copy(frame), source_type="camera", safe_name=None)

    @classmethod
    def from_file(cls, path: Path, *, source_type: SourceType = "file") -> ImageInput:
        """读取本地图片并转换为 OpenCV 使用的 BGR 数组。

        参数：
            path：待读取的图片路径。
            source_type：记录图片来源的标签。
        返回：
            不保留原始绝对路径的图片输入对象。
        前置条件：
            文件存在且能被 OpenCV 解码。
        """
        try:
            import cv2
        except ImportError as error:
            raise RuntimeError("OpenCV is not installed; install the project dependencies first") from error
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"could not decode image: {path.name}")
        return cls(frame=_validated_copy(frame), source_type=source_type, safe_name=path.name)


@dataclass(frozen=True)
class QualityProfile:
    """在人脸身份判定前使用的、可解释的图片质量数据。"""

    tier: Literal["high", "medium", "reject"]
    score: float
    metrics: dict[str, float]
    reasons: tuple[str, ...]


def assess_quality(
    frame: np.ndarray,
    observation: FaceObservation,
    settings: Settings,
) -> QualityProfile:
    """根据配置中的可复现指标评估检测到的人脸质量。

    参数：
        frame：原始 BGR 图像。
        observation：已经通过单人脸门控的人脸观察对象。
        settings：亮度、对比度和质量分层阈值。
    返回：
        包含质量分数、层级和拒绝原因的质量报告。
    前置条件：
        `observation.bbox` 必须对应当前图像中的有效区域。
    """

    crop = _crop_to_bbox(frame, observation.bbox)
    brightness = float(crop.mean())
    contrast = float(crop.std())
    left, top, right, bottom = observation.bbox
    face_size = min(right - left, bottom - top)
    metrics = {
        "detection_score": observation.detection_score,
        "face_size_px": face_size,
        "blur_variance": observation.blur_variance,
        "brightness": brightness,
        "contrast": contrast,
    }
    reasons = _hard_failure_reasons(metrics, settings)
    if reasons:
        return QualityProfile("reject", 0.0, metrics, tuple(reasons))

    score = _quality_score(metrics, settings)
    if score >= settings.high_quality_score:
        tier: Literal["high", "medium", "reject"] = "high"
    elif score >= settings.medium_quality_score:
        tier = "medium"
    else:
        tier = "reject"
        reasons = ["quality_score_below_medium"]
    return QualityProfile(tier, score, metrics, tuple(reasons))


def _validated_copy(frame: np.ndarray) -> np.ndarray:
    """验证并复制输入图像，确保后台处理拥有独立的连续数组。"""
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("image must be a three-channel BGR frame")
    if frame.size == 0:
        raise ValueError("image must not be empty")
    return np.ascontiguousarray(frame.copy())


def _crop_to_bbox(
    frame: np.ndarray,
    bbox: tuple[float, float, float, float],
) -> np.ndarray:
    """按检测框裁剪图像，并将坐标限制在图像边界内。"""
    height, width = frame.shape[:2]
    left = max(0, int(np.floor(bbox[0])))
    top = max(0, int(np.floor(bbox[1])))
    right = min(width, int(np.ceil(bbox[2])))
    bottom = min(height, int(np.ceil(bbox[3])))
    if right <= left or bottom <= top:
        return frame
    return frame[top:bottom, left:right]


def _hard_failure_reasons(metrics: dict[str, float], settings: Settings) -> list[str]:
    """返回会直接导致图片拒绝的质量指标原因。"""
    reasons: list[str] = []
    if metrics["detection_score"] < settings.min_detection_score:
        reasons.append("detection_score_below_minimum")
    if metrics["face_size_px"] < settings.min_face_size_px:
        reasons.append("face_size_below_minimum")
    if metrics["blur_variance"] < settings.min_blur_variance:
        reasons.append("blur_below_minimum")
    if metrics["brightness"] < settings.min_brightness:
        reasons.append("underexposed")
    if metrics["brightness"] > settings.max_brightness:
        reasons.append("overexposed")
    if metrics["contrast"] < settings.min_contrast:
        reasons.append("contrast_below_minimum")
    return reasons


def _quality_score(metrics: dict[str, float], settings: Settings) -> float:
    """将清晰度、亮度和对比度指标合成为 0 到 1 的质量分数。"""
    detection = _clamp(
        (metrics["detection_score"] - settings.min_detection_score)
        / (1.0 - settings.min_detection_score)
    )
    face_size = _clamp(metrics["face_size_px"] / (settings.min_face_size_px * 2))
    sharpness = _clamp(metrics["blur_variance"] / (settings.min_blur_variance * 2))
    exposure = _clamp(1.0 - abs(metrics["brightness"] - 127.5) / 127.5)
    contrast = _clamp(metrics["contrast"] / 64.0)
    return 0.25 * detection + 0.25 * face_size + 0.25 * sharpness + 0.15 * exposure + 0.10 * contrast


def _clamp(value: float) -> float:
    """把数值限制在闭区间 `[0, 1]`。"""
    return max(0.0, min(1.0, value))
