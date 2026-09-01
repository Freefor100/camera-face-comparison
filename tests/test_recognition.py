from __future__ import annotations

import hashlib

import numpy as np
import pytest

from camera_face_comparison.config import load_settings
from camera_face_comparison.face_engine import FaceObservation
from camera_face_comparison.image_input import ImageInput
from camera_face_comparison.recognition import (
    RecognitionService,
    aggregate_person_scores,
    aggregate_quality_weighted_scores,
    decide_match,
    recognize_embedding,
)
from camera_face_comparison.repository import FaceRepository, SampleInput


def test_aggregate_person_scores_uses_mean_of_two_best_samples() -> None:
    """A lower-quality third sample must not distort a person's final score."""

    person_scores = {
        "alice": [0.95, 0.72, 0.11],
        "bob": [0.85, 0.80],
    }

    assert aggregate_person_scores(person_scores) == {
        "alice": 0.835,
        "bob": 0.825,
    }


def test_quality_weighted_aggregation_downweights_a_poor_reference_image() -> None:
    """A very similar but low-quality reference must not dominate a reliable person."""

    aggregated = aggregate_quality_weighted_scores(
        {
            "alice": [(0.99, 0.10), (0.70, 0.95)],
            "bob": [(0.84, 0.95), (0.84, 0.95)],
        },
        top_k=2,
    )

    assert aggregated["alice"] == pytest.approx(0.8046, abs=0.0001)
    assert aggregated["bob"] == pytest.approx(0.84)
    assert aggregated["bob"] > aggregated["alice"]


def test_decide_match_returns_best_person_when_score_and_gap_pass() -> None:
    """Removing either acceptance check must prevent a false positive."""

    decision = decide_match(
        {"alice": 0.84, "bob": 0.70},
        match_threshold=0.80,
        min_margin=0.10,
    )

    assert decision.status == "matched"
    assert decision.person_id == "alice"
    assert decision.top_score == 0.84
    assert decision.runner_up_score == 0.70


def test_decide_match_returns_unknown_when_best_person_is_too_close() -> None:
    """A high score alone must not identify a person when candidates are ambiguous."""

    decision = decide_match(
        {"alice": 0.91, "bob": 0.87},
        match_threshold=0.80,
        min_margin=0.10,
    )

    assert decision.status == "unknown"
    assert decision.person_id is None
    assert decision.reason == "candidate_gap_below_minimum"


def test_recognize_embedding_uses_all_samples_before_selecting_person() -> None:
    """One excellent but unrepresentative sample must not beat a consistent identity."""

    decision = recognize_embedding(
        query_embedding=np.array([1.0, 0.0], dtype=np.float32),
        embeddings_by_person={
            "alice": [
                np.array([1.0, 0.0], dtype=np.float32),
                np.array([0.10, 0.995], dtype=np.float32),
            ],
            "bob": [
                np.array([0.91, 0.414], dtype=np.float32),
                np.array([0.90, 0.436], dtype=np.float32),
            ],
        },
        match_threshold=0.80,
        min_margin=0.05,
    )

    assert decision.status == "matched"
    assert decision.person_id == "bob"


def test_recognition_service_returns_name_for_best_library_identity(tmp_path) -> None:
    """The UI service must translate a matched person ID into the display name."""

    settings = load_settings(tmp_path)
    repository = FaceRepository(settings.database_path)
    alice = _create_person(
        repository,
        settings,
        "Alice",
        [np.array([1.0, 0.0], dtype=np.float32)],
    )
    _create_person(
        repository,
        settings,
        "Bob",
        [np.array([0.2, 0.98], dtype=np.float32)],
    )

    class ProbeEngine:
        def extract_single_face(self, frame: np.ndarray) -> FaceObservation:
            return FaceObservation(
                bbox=(0.0, 0.0, 160.0, 160.0),
                detection_score=0.95,
                embedding=np.array([1.0, 0.0], dtype=np.float32),
                blur_variance=150.0,
                landmarks=None,
            )

    result = RecognitionService(repository, settings, ProbeEngine()).compare(
        _textured_frame(120)
    )
    repository.close()

    assert result.status == "matched"
    assert result.display_name == "Alice"
    assert result.person_id == alice.id
    assert result.bbox == (0.0, 0.0, 160.0, 160.0)


def test_recognition_uses_stricter_policy_for_medium_quality_probe(tmp_path) -> None:
    """The same similarity is accepted for a high-quality probe but rejected for medium quality."""

    settings = load_settings(tmp_path)
    repository = FaceRepository(settings.database_path)
    _create_person(
        repository,
        settings,
        "Alice",
        [np.array([1.0, 0.0], dtype=np.float32)] * 3,
    )

    class ProbeEngine:
        def extract_single_face(self, frame: np.ndarray) -> FaceObservation:
            return FaceObservation(
                bbox=(0.0, 0.0, 180.0, 180.0),
                detection_score=0.95,
                embedding=np.array([0.55, 0.83516465], dtype=np.float32),
                blur_variance=150.0,
                landmarks=None,
            )

    service = RecognitionService(repository, settings, ProbeEngine())
    high_quality = _textured_frame(120)
    medium_quality = _textured_frame(50)

    high_result = service.compare_input(ImageInput.from_camera(high_quality))
    medium_result = service.compare_input(ImageInput.from_camera(medium_quality))
    repository.close()

    assert high_result.status == "matched"
    assert medium_result.status == "unknown"
    assert medium_result.reason == "score_below_threshold"


def _create_person(
    repository: FaceRepository,
    settings,
    name: str,
    embeddings: list[np.ndarray],
):
    person_id = name.lower()
    samples = []
    for index, embedding in enumerate(embeddings, start=1):
        relative_path = f"faces/{person_id}/sample_{index:03d}.jpg"
        image_path = settings.data_dir / relative_path
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(f"{name}-{index}".encode())
        samples.append(
            SampleInput(
                image_path=relative_path,
                embedding=embedding,
                pose=f"sample_{index:03d}",
                quality={"quality_score": 0.95, "tier": "high"},
                source_type="file",
                image_sha256=hashlib.sha256(image_path.read_bytes()).hexdigest(),
            )
        )
    return repository.create_person_with_samples(
        person_id=person_id,
        display_name=name,
        samples=samples,
    )


def _textured_frame(base_value: int) -> np.ndarray:
    frame = np.full((240, 320, 3), base_value, dtype=np.uint8)
    frame[:, ::2] = base_value + 30
    return frame
