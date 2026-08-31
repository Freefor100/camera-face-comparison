from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

import numpy as np

from .config import Settings
from .domain import RecognitionResult
from .face_engine import FaceInputError, FaceObservation
from .repository import FaceRepository


@dataclass(frozen=True)
class MatchDecision:
    """A recognition decision made from a set of person-level scores."""

    status: str
    person_id: str | None
    top_score: float | None
    runner_up_score: float | None
    reason: str | None


class ProbeFaceEngine(Protocol):
    def extract_single_face(self, frame: np.ndarray) -> FaceObservation: ...


class RecognitionService:
    """Application service that joins a probe face with the persisted library."""

    def __init__(
        self,
        repository: FaceRepository,
        settings: Settings,
        face_engine: ProbeFaceEngine,
    ) -> None:
        self._repository = repository
        self._settings = settings
        self._face_engine = face_engine

    def compare(self, frame: np.ndarray) -> RecognitionResult:
        started_at = perf_counter()
        try:
            probe = self._face_engine.extract_single_face(frame)
            people = self._repository.list_people()
            samples = self._repository.list_samples()
            embeddings_by_person: dict[str, list[np.ndarray]] = {}
            for sample in samples:
                embeddings_by_person.setdefault(sample.person_id, []).append(sample.embedding)
            decision = recognize_embedding(
                query_embedding=probe.embedding,
                embeddings_by_person=embeddings_by_person,
                match_threshold=self._settings.match_threshold,
                min_margin=self._settings.min_margin,
            )
            names = {person.id: person.display_name for person in people}
            result = RecognitionResult(
                status=decision.status,
                person_id=decision.person_id,
                display_name=names.get(decision.person_id),
                top_score=decision.top_score,
                runner_up_score=decision.runner_up_score,
                latency_ms=(perf_counter() - started_at) * 1000,
                reason=decision.reason,
                bbox=probe.bbox,
            )
        except FaceInputError as error:
            result = RecognitionResult(
                status="invalid",
                person_id=None,
                display_name=None,
                top_score=None,
                runner_up_score=None,
                latency_ms=(perf_counter() - started_at) * 1000,
                reason=str(error),
                bbox=None,
            )
        self._repository.record_recognition(
            decision=result.status,
            person_id=result.person_id,
            top_score=result.top_score,
            runner_up_score=result.runner_up_score,
            latency_ms=result.latency_ms,
            reason=result.reason,
        )
        return result


def aggregate_person_scores(
    person_scores: Mapping[str, Sequence[float]],
) -> dict[str, float]:
    """Return each person's mean score across their two best samples."""

    aggregated: dict[str, float] = {}
    for person_id, scores in person_scores.items():
        best_scores = sorted(scores, reverse=True)[:2]
        if best_scores:
            aggregated[person_id] = sum(best_scores) / len(best_scores)
    return aggregated


def decide_match(
    person_scores: Mapping[str, float],
    *,
    match_threshold: float,
    min_margin: float,
) -> MatchDecision:
    """Return a match only when the best person clears both safety checks."""

    ranked = sorted(person_scores.items(), key=lambda item: item[1], reverse=True)
    if not ranked:
        return MatchDecision("unknown", None, None, None, "empty_face_library")

    person_id, top_score = ranked[0]
    runner_up_score = ranked[1][1] if len(ranked) > 1 else None
    if top_score < match_threshold:
        return MatchDecision(
            "unknown",
            None,
            top_score,
            runner_up_score,
            "score_below_threshold",
        )
    if runner_up_score is not None and top_score - runner_up_score < min_margin:
        return MatchDecision(
            "unknown",
            None,
            top_score,
            runner_up_score,
            "candidate_gap_below_minimum",
        )
    return MatchDecision("matched", person_id, top_score, runner_up_score, None)


def recognize_embedding(
    *,
    query_embedding: np.ndarray,
    embeddings_by_person: Mapping[str, Sequence[np.ndarray]],
    match_threshold: float,
    min_margin: float,
) -> MatchDecision:
    """Compare one probe vector with every enrolled sample and decide safely."""

    query = _normalize(query_embedding)
    raw_scores: dict[str, list[float]] = {}
    for person_id, embeddings in embeddings_by_person.items():
        scores = [float(query @ _normalize(embedding)) for embedding in embeddings]
        if scores:
            raw_scores[person_id] = scores
    return decide_match(
        aggregate_person_scores(raw_scores),
        match_threshold=match_threshold,
        min_margin=min_margin,
    )


def _normalize(embedding: np.ndarray) -> np.ndarray:
    vector = np.asarray(embedding, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if vector.ndim != 1 or vector.size == 0 or norm == 0.0:
        raise ValueError("embedding must be a non-zero one-dimensional vector")
    return vector / norm
