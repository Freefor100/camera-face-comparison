from __future__ import annotations

import numpy as np

from camera_face_comparison.config import load_settings
from camera_face_comparison.face_engine import FaceObservation
from camera_face_comparison.recognition import (
    RecognitionService,
    aggregate_person_scores,
    decide_match,
    recognize_embedding,
)
from camera_face_comparison.repository import FaceRepository


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
    alice = repository.create_person("Alice")
    bob = repository.create_person("Bob")
    repository.add_sample(
        person_id=alice.id,
        image_path="faces/alice/front.jpg",
        embedding=np.array([1.0, 0.0], dtype=np.float32),
        pose="front",
        quality={},
    )
    repository.add_sample(
        person_id=alice.id,
        image_path="faces/alice/left.jpg",
        embedding=np.array([0.99, 0.1], dtype=np.float32),
        pose="left",
        quality={},
    )
    repository.add_sample(
        person_id=bob.id,
        image_path="faces/bob/front.jpg",
        embedding=np.array([0.2, 0.98], dtype=np.float32),
        pose="front",
        quality={},
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
        np.zeros((100, 100, 3), dtype=np.uint8)
    )
    repository.close()

    assert result.status == "matched"
    assert result.display_name == "Alice"
    assert result.person_id == alice.id
    assert result.bbox == (0.0, 0.0, 160.0, 160.0)
