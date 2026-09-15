from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
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


@dataclass(frozen=True)
class DetectedFace:
    """尚未执行身份特征提取的人脸检测结果。"""

    bbox: tuple[float, float, float, float]
    detection_score: float
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
        """检测图片中的全部人脸，并逐张提取归一化身份特征。

        参数：
            frame：OpenCV 读取的 BGR 图像。
        返回：
            包含框、检测分数、特征、关键点和清晰度的观察对象列表。
        """

        return [self.extract_detected_face(frame, face) for face in self.detect_faces(frame)]

    def detect_faces(self, frame: np.ndarray) -> list[DetectedFace]:
        """只执行人脸检测，不运行身份特征模型。

        参数：
            frame：OpenCV 读取的 BGR 图像。
        返回：
            人脸框、检测分、五点关键点和人脸区域清晰度组成的检测结果。
        前置条件：
            InsightFace 分析器已经加载并准备好 detection 模块。
        """

        detector = getattr(self._analyzer, "det_model", None)
        if detector is None:
            raise RuntimeError("InsightFace detection model is unavailable")
        bboxes, keypoints = detector.detect(frame, max_num=0, metric="default")
        detected: list[DetectedFace] = []
        for index, row in enumerate(np.asarray(bboxes)):
            if row.size < 5:
                continue
            bbox = tuple(float(value) for value in row[:4])
            normalized_bbox = (bbox[0], bbox[1], bbox[2], bbox[3])
            landmarks = None
            if keypoints is not None:
                landmarks = np.asarray(keypoints[index], dtype=np.float32)
            detected.append(
                DetectedFace(
                    bbox=normalized_bbox,
                    detection_score=float(row[4]),
                    blur_variance=self._blur_metric(_face_crop(frame, normalized_bbox)),
                    landmarks=landmarks,
                )
            )
        return detected

    def detect_single_face(self, frame: np.ndarray) -> DetectedFace:
        """只执行检测并要求图片中恰好存在一张人脸。"""

        faces = self.detect_faces(frame)
        if not faces:
            raise FaceInputError("no_face_detected")
        if len(faces) != 1:
            raise FaceInputError("multiple_faces")
        return faces[0]

    def extract_detected_face(
        self,
        frame: np.ndarray,
        detected_face: DetectedFace,
    ) -> FaceObservation:
        """对一张已检测人脸执行五点对齐和身份特征提取。

        参数：
            frame：检测结果所属的原始 BGR 图像。
            detected_face：`detect_faces` 返回的一张人脸。
        返回：
            包含 L2 单位身份特征向量的完整人脸观察。
        前置条件：
            检测结果必须包含识别模型对齐所需的五点关键点。
        """

        if detected_face.landmarks is None:
            raise FaceInputError("missing_face_landmarks")
        models = getattr(self._analyzer, "models", None)
        recognizer = models.get("recognition") if isinstance(models, dict) else None
        if recognizer is None:
            raise RuntimeError("InsightFace recognition model is unavailable")
        vendor_face = SimpleNamespace(
            bbox=np.asarray(detected_face.bbox, dtype=np.float32),
            det_score=detected_face.detection_score,
            kps=np.asarray(detected_face.landmarks, dtype=np.float32),
        )
        embedding = recognizer.get(frame, vendor_face)
        return FaceObservation(
            bbox=detected_face.bbox,
            detection_score=detected_face.detection_score,
            embedding=normalize_embedding(np.asarray(embedding, dtype=np.float32)),
            blur_variance=detected_face.blur_variance,
            landmarks=detected_face.landmarks,
        )

    def extract_single_face(self, frame: np.ndarray) -> FaceObservation:
        """依次执行单脸检测和身份特征提取。"""

        return self.extract_detected_face(frame, self.detect_single_face(frame))


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
