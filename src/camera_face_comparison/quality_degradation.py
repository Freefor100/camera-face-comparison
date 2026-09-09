from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .face_engine import FaceInputError, FaceObservation, normalize_embedding

DegradationKind = Literal[
    "baseline",
    "face_size",
    "gaussian_blur",
    "brightness",
    "contrast",
]


@dataclass(frozen=True)
class DegradationSpec:
    """一个只改变单项图像因素的确定性退化条件。"""

    kind: DegradationKind
    level: float

    @property
    def key(self) -> str:
        """返回可作为 SQLite 主键组成部分的稳定条件名称。"""

        return f"{self.kind}:{self.level:g}"


def degradation_specs() -> tuple[DegradationSpec, ...]:
    """返回去重后的 19 个基线或单因素退化条件。"""

    return (
        DegradationSpec("baseline", 1.0),
        *(DegradationSpec("face_size", value) for value in (160, 112, 96, 80, 64, 48)),
        *(DegradationSpec("gaussian_blur", value) for value in (1, 2, 3, 4)),
        *(
            DegradationSpec("brightness", value)
            for value in (0.75, 0.50, 0.35, 1.25, 1.50)
        ),
        *(DegradationSpec("contrast", value) for value in (0.75, 0.50, 0.25)),
    )


def apply_degradation(
    frame: np.ndarray,
    spec: DegradationSpec,
    baseline_bbox: tuple[float, float, float, float] | None,
) -> np.ndarray:
    """按实验条件返回一张新 BGR 图像，不修改输入数组。

    参数：
        frame：原始三通道 BGR 图片。
        spec：基线或一个单因素退化条件。
        baseline_bbox：原图主脸检测框，仅人脸尺寸条件需要。
    返回：
        可直接交给人脸模型的连续 uint8 图像。
    """

    if frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0:
        raise ValueError("frame must be a non-empty three-channel image")
    if spec.kind == "baseline":
        return np.ascontiguousarray(frame.copy())
    if spec.kind == "brightness":
        return _as_uint8(frame.astype(np.float32) * spec.level)
    if spec.kind == "contrast":
        pixels = frame.astype(np.float32)
        mean = float(pixels.mean())
        return _as_uint8(mean + (pixels - mean) * spec.level)
    if spec.kind == "gaussian_blur":
        if spec.level <= 0:
            raise ValueError("gaussian blur sigma must be positive")
        try:
            import cv2
        except ImportError as error:
            raise RuntimeError("OpenCV is required for quality degradation") from error
        radius = max(1, int(np.ceil(3 * spec.level)))
        kernel_size = radius * 2 + 1
        return np.ascontiguousarray(
            cv2.GaussianBlur(frame, (kernel_size, kernel_size), spec.level)
        )
    if spec.kind == "face_size":
        if baseline_bbox is None:
            raise ValueError("baseline_bbox is required for face_size degradation")
        return _place_scaled_on_canvas(frame, baseline_bbox, spec.level)
    raise ValueError(f"unsupported degradation kind: {spec.kind}")


def select_primary_face(faces: Sequence[FaceObservation]) -> FaceObservation:
    """从单主体数据集检测结果中确定面积最大的主脸并归一化特征。

    该规则只用于已有身份标签且画面主体明确的数据集实验；桌面应用仍执行多人脸拒绝。
    """

    if not faces:
        raise FaceInputError("no_face_detected")
    selected = max(
        faces,
        key=lambda face: (
            _bbox_area(face.bbox),
            face.detection_score,
            tuple(-value for value in face.bbox),
        ),
    )
    return FaceObservation(
        bbox=selected.bbox,
        detection_score=selected.detection_score,
        embedding=normalize_embedding(selected.embedding),
        blur_variance=selected.blur_variance,
        landmarks=selected.landmarks,
    )


def _place_scaled_on_canvas(
    frame: np.ndarray,
    bbox: tuple[float, float, float, float],
    target_face_size: float,
) -> np.ndarray:
    """缩放整张图并居中放到 640 像素灰色画布，使主脸接近目标尺寸。"""

    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("OpenCV is required for quality degradation") from error
    left, top, right, bottom = bbox
    source_face_size = min(right - left, bottom - top)
    if source_face_size <= 0 or target_face_size <= 0:
        raise ValueError("face sizes must be positive")
    scale = target_face_size / source_face_size
    resized_width = max(1, round(frame.shape[1] * scale))
    resized_height = max(1, round(frame.shape[0] * scale))
    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)

    canvas_size = 640
    canvas = np.full((canvas_size, canvas_size, 3), 127, dtype=np.uint8)
    source_x = max(0, (resized_width - canvas_size) // 2)
    source_y = max(0, (resized_height - canvas_size) // 2)
    target_x = max(0, (canvas_size - resized_width) // 2)
    target_y = max(0, (canvas_size - resized_height) // 2)
    copy_width = min(resized_width, canvas_size)
    copy_height = min(resized_height, canvas_size)
    canvas[target_y : target_y + copy_height, target_x : target_x + copy_width] = resized[
        source_y : source_y + copy_height,
        source_x : source_x + copy_width,
    ]
    return np.ascontiguousarray(canvas)


def _bbox_area(bbox: tuple[float, float, float, float]) -> float:
    """返回非负检测框面积。"""

    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _as_uint8(values: np.ndarray) -> np.ndarray:
    """把浮点像素裁剪为连续 uint8 图像。"""

    return np.ascontiguousarray(np.clip(np.rint(values), 0, 255).astype(np.uint8))
