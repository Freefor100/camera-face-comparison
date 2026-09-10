from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import Settings
from .runtime import ExecutionBackend, actual_backend_for_analyzer, detect_execution_backend


class FaceInputError(ValueError):
    """图片无法产生唯一且有效的人脸特征。"""


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
        *,
        analyzer: Any,
        blur_metric: Callable[[np.ndarray], float] | None = None,
        backend: ExecutionBackend | None = None,
    ) -> None:
        """保存模型分析器和质量测量函数。

        参数：
            analyzer：提供 `get(frame)` 方法的 InsightFace 兼容分析器。
            blur_metric：可选的清晰度计算函数，未提供时使用拉普拉斯方差。
            backend：创建分析器时使用的推理后端；直接注入分析器的测试默认按 CPU 标记。
        """
        self._analyzer = analyzer
        self._blur_metric = blur_metric or _laplacian_variance
        self._backend = backend or ExecutionBackend(
            name="cpu", providers=("CPUExecutionProvider",), context_id=-1
        )

    @classmethod
    def from_local_model(cls, settings: Settings) -> FaceEngine:
        """只从数据目录加载本地 InsightFace 模型，不允许启动时联网下载。

        参数：
            settings：包含模型目录和日志目录的运行配置。
        返回：
            已选择可用推理后端并准备好的模型适配器。
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
        backend = detect_execution_backend()
        try:
            from insightface.app import FaceAnalysis
        except ImportError as error:
            raise RuntimeError(
                "InsightFace is not installed; install the project dependencies first"
            ) from error
        analyzer = FaceAnalysis(
            name="buffalo_l",
            root=str(settings.data_dir),
            providers=list(backend.providers),
            allowed_modules=["detection", "recognition"],
        )
        analyzer.prepare(ctx_id=backend.context_id, det_size=(640, 640))
        actual_backend = actual_backend_for_analyzer(analyzer, backend)
        return cls(analyzer=analyzer, backend=actual_backend)

    @property
    def backend(self) -> ExecutionBackend:
        """返回当前模型适配器实际选择的推理后端。"""

        return self._backend

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
        """要求图中恰好一张人脸，并返回归一化 embedding。"""

        return validate_single_face(self.extract_faces(frame))


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """返回适合余弦相似度计算的 float32 单位向量。"""

    vector = np.asarray(embedding, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if vector.ndim != 1 or vector.size == 0 or norm == 0.0:
        raise FaceInputError("invalid_embedding")
    return vector / norm


def validate_single_face(faces: Sequence[FaceObservation]) -> FaceObservation:
    """检查人脸数量和 embedding 有效性，不使用数值质量硬门。

    参数：
        faces：当前图像中检测到的人脸观察对象。
    返回：
        唯一一张人脸，其特征已完成 L2 归一化。
    前置条件：
        输入来自同一图像；无人脸、多张脸或无效 embedding 时抛出 `FaceInputError`。
    """

    if not faces:
        raise FaceInputError("no_face_detected")
    if len(faces) != 1:
        raise FaceInputError("multiple_faces")
    face = faces[0]
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
