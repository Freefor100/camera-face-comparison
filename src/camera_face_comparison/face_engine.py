from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import Settings


class FaceInputError(ValueError):
    """摄像头帧不能安全用于录入或识别。"""


@dataclass(frozen=True)
class FaceObservation:
    """一张检测到的人脸，使用与具体模型厂商无关的数据结构表示。"""

    bbox: tuple[float, float, float, float]
    detection_score: float
    embedding: np.ndarray
    blur_variance: float
    landmarks: np.ndarray | None


class FaceEngine:
    """隔离 InsightFace 对象与应用服务的模型适配器。"""

    def __init__(
        self,
        settings: Settings,
        *,
        analyzer: Any,
        blur_metric: Callable[[np.ndarray], float] | None = None,
    ) -> None:
        """保存模型分析器和质量度量函数。

        参数：
            settings：当前质量门控配置。
            analyzer：提供 `get(frame)` 方法的 InsightFace 兼容分析器。
            blur_metric：可选的清晰度计算函数，未提供时使用拉普拉斯方差。
        """
        self._settings = settings
        self._analyzer = analyzer
        self._blur_metric = blur_metric or _laplacian_variance

    @classmethod
    def from_local_model(cls, settings: Settings) -> FaceEngine:
        """只从数据目录加载本地 InsightFace 模型，不允许启动时联网下载。

        参数：
            settings：包含模型目录和日志目录的运行配置。
        返回：
            已使用 CPU 推理提供器准备好的模型适配器。
        前置条件：
            `data/models/buffalo_l` 必须已经存在，依赖包也必须已安装。
        """

        model_dir = settings.models_dir / "buffalo_l"
        if not model_dir.is_dir():
            raise RuntimeError(
                f"offline model is missing at {model_dir}; run scripts/prepare_models.py first"
            )
        os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
        os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
        matplotlib_cache = settings.logs_dir / "matplotlib"
        matplotlib_cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
        try:
            from insightface.app import FaceAnalysis
        except ImportError as error:
            raise RuntimeError(
                "InsightFace is not installed; install the project dependencies first"
            ) from error
        analyzer = FaceAnalysis(
            name="buffalo_l",
            root=str(settings.data_dir),
            providers=["CPUExecutionProvider"],
        )
        analyzer.prepare(ctx_id=-1, det_size=(640, 640))
        return cls(settings, analyzer=analyzer)

    def extract_faces(self, frame: np.ndarray) -> list[FaceObservation]:
        """把一张 BGR 图像中的所有检测结果转换为通用人脸观察对象。

        参数：
            frame：OpenCV 读取的 BGR 图像。
        返回：
            包含框、检测分数、特征、关键点和清晰度的观察对象列表。
        """

        observations: list[FaceObservation] = []
        for face in self._analyzer.get(frame):
            bbox = tuple(float(value) for value in face.bbox)
            if len(bbox) != 4:
                continue
            normalized_bbox = (bbox[0], bbox[1], bbox[2], bbox[3])
            landmarks = getattr(face, "kps", None)
            observations.append(
                FaceObservation(
                    bbox=normalized_bbox,
                    detection_score=float(face.det_score),
                    embedding=np.asarray(face.embedding, dtype=np.float32),
                    blur_variance=self._blur_metric(_face_crop(frame, normalized_bbox)),
                    landmarks=(
                        np.asarray(landmarks, dtype=np.float32) if landmarks is not None else None
                    ),
                )
            )
        return observations

    def extract_single_face(self, frame: np.ndarray) -> FaceObservation:
        """提取并返回一张通过质量门控的归一化人脸。"""

        return validate_single_face(self.extract_faces(frame), self._settings)


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """返回适合余弦相似度计算的 float32 单位向量。"""

    vector = np.asarray(embedding, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if vector.ndim != 1 or vector.size == 0 or norm == 0.0:
        raise FaceInputError("invalid_embedding")
    return vector / norm


def validate_single_face(
    faces: Sequence[FaceObservation], settings: Settings
) -> FaceObservation:
    """在人脸数据进入标准库前执行单人脸、尺寸、分数和清晰度检查。

    参数：
        faces：当前图像中检测到的人脸观察对象。
        settings：质量门控阈值。
    返回：
        一张通过检查且特征已归一化的人脸观察对象。
    前置条件：
        输入应来自同一帧图像；无人脸、多张有效人脸或低质量时抛出 `FaceInputError`。
    """

    if not faces:
        raise FaceInputError("no_face_detected")
    accepted_detections = [
        face for face in faces if face.detection_score >= settings.min_detection_score
    ]
    if not accepted_detections:
        raise FaceInputError("detection_score_below_minimum")
    if len(accepted_detections) != 1:
        raise FaceInputError("multiple_faces")
    face = accepted_detections[0]
    left, top, right, bottom = face.bbox
    face_size = min(right - left, bottom - top)
    if face_size < settings.min_face_size_px:
        raise FaceInputError("face_size_below_minimum")
    if face.blur_variance < settings.min_blur_variance:
        raise FaceInputError("blur_below_minimum")
    normalized_embedding = normalize_embedding(face.embedding)
    return FaceObservation(
        bbox=face.bbox,
        detection_score=face.detection_score,
        embedding=normalized_embedding,
        blur_variance=face.blur_variance,
        landmarks=face.landmarks,
    )


def _laplacian_variance(frame: np.ndarray) -> float:
    """使用 OpenCV 拉普拉斯算子的方差估计图像清晰度。"""
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("OpenCV is not installed; install the project dependencies first") from error
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(grayscale, cv2.CV_64F).var())


def _face_crop(
    frame: np.ndarray,
    bbox: tuple[float, float, float, float],
) -> np.ndarray:
    """裁剪检测框区域用于清晰度计算，并容忍越过图像边界的检测框。"""

    height, width = frame.shape[:2]
    left = max(0, int(np.floor(bbox[0])))
    top = max(0, int(np.floor(bbox[1])))
    right = min(width, int(np.ceil(bbox[2])))
    bottom = min(height, int(np.ceil(bbox[3])))
    if right <= left or bottom <= top:
        return frame
    return frame[top:bottom, left:right]
