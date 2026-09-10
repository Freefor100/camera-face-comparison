from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from .config import QualityWarningThresholds

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


def measure_quality(
    frame: np.ndarray,
    observation: FaceObservation,
) -> dict[str, float]:
    """返回不依赖任何质量阈值的原始人脸指标。

    参数：
        frame：原始 BGR 图像。
        observation：模型在当前图像中产生的人脸观察对象。
    返回：
        检测置信度、人脸尺寸、清晰度、亮度和对比度组成的数值字典。
    前置条件：
        `observation.bbox` 必须对应当前图像中的有效区域。
    """

    crop = _crop_to_bbox(frame, observation.bbox)
    left, top, right, bottom = observation.bbox
    return {
        "detection_score": observation.detection_score,
        "face_size_px": min(right - left, bottom - top),
        "blur_variance": observation.blur_variance,
        "brightness": float(crop.mean()),
        "contrast": float(crop.std()),
    }


def quality_warnings(
    metrics: dict[str, float],
    thresholds: QualityWarningThresholds,
) -> tuple[str, ...]:
    """把异常质量指标转换为不会阻断录入或识别的操作提示。

    参数：
        metrics：`measure_quality()` 产生的五项原始指标。
        thresholds：只用于生成提示的参考界限。
    返回：
        稳定的提示代码；空元组表示当前指标没有明显问题。
    """

    warnings: list[str] = []
    if metrics["detection_score"] < thresholds.low_detection_score:
        warnings.append("low_detection_confidence")
    if metrics["face_size_px"] < thresholds.small_face_size_px:
        warnings.append("move_closer")
    if metrics["blur_variance"] < thresholds.low_blur_variance:
        warnings.append("hold_still")
    if metrics["brightness"] < thresholds.low_brightness:
        warnings.append("increase_lighting")
    elif metrics["brightness"] > thresholds.high_brightness:
        warnings.append("reduce_lighting")
    if metrics["contrast"] < thresholds.low_contrast:
        warnings.append("improve_contrast")
    return tuple(warnings)


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
