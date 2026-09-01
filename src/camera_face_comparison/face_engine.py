from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import Settings


class FaceInputError(ValueError):
    """A camera frame cannot safely be used for enrollment or recognition."""


@dataclass(frozen=True)
class FaceObservation:
    """One detected face, represented independently from a model vendor."""

    bbox: tuple[float, float, float, float]
    detection_score: float
    embedding: np.ndarray
    blur_variance: float
    landmarks: np.ndarray | None


class FaceEngine:
    """Adapter that keeps InsightFace-specific objects outside application services."""

    def __init__(
        self,
        settings: Settings,
        *,
        analyzer: Any,
        blur_metric: Callable[[np.ndarray], float] | None = None,
    ) -> None:
        self._settings = settings
        self._analyzer = analyzer
        self._blur_metric = blur_metric or _laplacian_variance

    @classmethod
    def from_local_model(cls, settings: Settings) -> FaceEngine:
        """Load the local InsightFace model without allowing a network download."""

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
        """Convert all detections in one BGR frame to vendor-neutral observations."""

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
        """Return a quality-approved, normalized face for one capture frame."""

        return validate_single_face(self.extract_faces(frame), self._settings)


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """Return a float32 unit vector suitable for cosine-similarity scoring."""

    vector = np.asarray(embedding, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if vector.ndim != 1 or vector.size == 0 or norm == 0.0:
        raise FaceInputError("invalid_embedding")
    return vector / norm


def validate_single_face(
    faces: Sequence[FaceObservation], settings: Settings
) -> FaceObservation:
    """Enforce the local quality policy before face data enters the library."""

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
    """Crop the detected face before measuring blur, while tolerating edge boxes."""

    height, width = frame.shape[:2]
    left = max(0, int(np.floor(bbox[0])))
    top = max(0, int(np.floor(bbox[1])))
    right = min(width, int(np.ceil(bbox[2])))
    bottom = min(height, int(np.ceil(bbox[3])))
    if right <= left or bottom <= top:
        return frame
    return frame[top:bottom, left:right]
