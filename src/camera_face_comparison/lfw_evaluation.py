from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

import numpy as np

from .config import Settings
from .experiment import ExperimentRecord
from .face_engine import FaceInputError, FaceObservation, validate_single_face
from .image_input import ImageInput, QualityProfile, assess_quality
from .lfw_dataset import LfwProtocol


class EvaluationFaceEngine(Protocol):
    def extract_single_face(self, frame: np.ndarray) -> FaceObservation: ...


@dataclass(frozen=True)
class DatasetRejection:
    """One LFW image excluded before scoring, with a reproducible reason."""

    relative_path: str
    reason: str


@dataclass(frozen=True)
class LfwEvaluationRun:
    """Usable scores and explicit exclusions from one fixed LFW protocol."""

    active_person_ids: tuple[str, ...]
    records: tuple[ExperimentRecord, ...]
    enrollment_rejections: tuple[DatasetRejection, ...]
    probe_rejections: tuple[DatasetRejection, ...]


def evaluate_lfw_protocol(
    *,
    dataset_dir,
    protocol: LfwProtocol,
    settings: Settings,
    face_engine: EvaluationFaceEngine,
) -> LfwEvaluationRun:
    """Extract LFW embeddings and return records for the same open-set scoring code as the app."""

    gallery_embeddings: dict[str, list[np.ndarray]] = {}
    gallery_quality: dict[str, list[float]] = {}
    enrollment_rejections: list[DatasetRejection] = []
    for person_id, paths in protocol.enrollment.items():
        person_embeddings: list[np.ndarray] = []
        person_quality: list[float] = []
        for relative_path in paths:
            try:
                embedding, profile = _extract_valid_embedding(
                    dataset_dir / relative_path, settings, face_engine
                )
            except (FaceInputError, ValueError) as error:
                enrollment_rejections.append(DatasetRejection(relative_path, str(error)))
                continue
            person_embeddings.append(embedding)
            person_quality.append(profile.score)
        if len(person_embeddings) >= settings.min_active_samples:
            gallery_embeddings[person_id] = person_embeddings
            gallery_quality[person_id] = person_quality

    records: list[ExperimentRecord] = []
    probe_rejections: list[DatasetRejection] = []
    for probe in protocol.probes:
        started_at = perf_counter()
        try:
            query, profile = _extract_valid_embedding(
                dataset_dir / probe.relative_path, settings, face_engine
            )
        except (FaceInputError, ValueError) as error:
            probe_rejections.append(DatasetRejection(probe.relative_path, str(error)))
            continue
        sample_scores = {
            person_id: [float(query @ embedding) for embedding in embeddings]
            for person_id, embeddings in gallery_embeddings.items()
        }
        records.append(
            ExperimentRecord(
                expected_person_id=probe.expected_person_id,
                sample_scores=sample_scores,
                latency_ms=(perf_counter() - started_at) * 1000,
                sample_quality_scores=gallery_quality,
                probe_quality_tier=profile.tier,
            )
        )
    return LfwEvaluationRun(
        active_person_ids=tuple(gallery_embeddings),
        records=tuple(records),
        enrollment_rejections=tuple(enrollment_rejections),
        probe_rejections=tuple(probe_rejections),
    )


def _extract_valid_embedding(
    image_path,
    settings: Settings,
    face_engine: EvaluationFaceEngine,
) -> tuple[np.ndarray, QualityProfile]:
    image_input = ImageInput.from_file(image_path, source_type="dataset")
    observed_face = face_engine.extract_single_face(image_input.frame)
    face = validate_single_face([observed_face], settings)
    profile = assess_quality(image_input.frame, face, settings)
    if profile.tier == "reject":
        raise FaceInputError("quality_rejected:" + ",".join(profile.reasons or ("low_score",)))
    return face.embedding, profile
