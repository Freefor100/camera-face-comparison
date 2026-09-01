from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

import numpy as np

from .config import Settings
from .domain import RecognitionResult
from .face_engine import FaceInputError, FaceObservation, validate_single_face
from .image_input import ImageInput, assess_quality
from .integrity import verify_library
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
        """Compare a live camera frame through the same flow as an imported image."""

        return self.compare_input(ImageInput.from_camera(frame))

    def compare_input(self, image_input: ImageInput) -> RecognitionResult:
        """Perform quality-aware open-set 1:N identification for one local image."""

        started_at = perf_counter()
        try:
            verification = verify_library(self._repository, self._settings)
            if not verification.is_valid:
                first_failure = verification.failures[0]
                result = RecognitionResult(
                    status="invalid",
                    person_id=None,
                    display_name=None,
                    top_score=None,
                    runner_up_score=None,
                    latency_ms=(perf_counter() - started_at) * 1000,
                    reason=f"library_integrity_failed:{first_failure.kind}",
                    bbox=None,
                )
                return self._record_and_return(result)

            observed_probe = self._face_engine.extract_single_face(image_input.frame)
            probe = validate_single_face([observed_probe], self._settings)
            profile = assess_quality(image_input.frame, probe, self._settings)
            if profile.tier == "reject":
                result = RecognitionResult(
                    status="invalid",
                    person_id=None,
                    display_name=None,
                    top_score=None,
                    runner_up_score=None,
                    latency_ms=(perf_counter() - started_at) * 1000,
                    reason="quality_rejected:" + ",".join(profile.reasons or ("low_score",)),
                    bbox=probe.bbox,
                )
                return self._record_and_return(result)

            people = self._repository.list_people()
            samples = self._repository.list_samples()
            embeddings_by_person: dict[str, list[np.ndarray]] = {}
            quality_by_person: dict[str, list[float]] = {}
            for sample in samples:
                embeddings_by_person.setdefault(sample.person_id, []).append(sample.embedding)
                quality_by_person.setdefault(sample.person_id, []).append(
                    _stored_quality_score(sample.quality)
                )
            policy = self._settings.quality_tiers[profile.tier]
            decision = recognize_embedding(
                query_embedding=probe.embedding,
                embeddings_by_person=embeddings_by_person,
                sample_quality_by_person=quality_by_person,
                top_k=self._settings.top_k,
                match_threshold=policy.match_threshold,
                min_margin=policy.min_margin,
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
        except (FaceInputError, TypeError, ValueError) as error:
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
        return self._record_and_return(result)

    def _record_and_return(self, result: RecognitionResult) -> RecognitionResult:
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
    *,
    top_k: int = 2,
) -> dict[str, float]:
    """Return each person's mean score across its strongest reference samples."""

    if top_k < 1:
        raise ValueError("top_k must be at least one")
    aggregated: dict[str, float] = {}
    for person_id, scores in person_scores.items():
        best_scores = sorted(scores, reverse=True)[:top_k]
        if best_scores:
            aggregated[person_id] = sum(best_scores) / len(best_scores)
    return aggregated


def aggregate_quality_weighted_scores(
    person_scores: Mapping[str, Sequence[tuple[float, float]]],
    *,
    top_k: int,
) -> dict[str, float]:
    """Aggregate Top-K cosine similarities while reducing low-quality reference influence."""

    if top_k < 1:
        raise ValueError("top_k must be at least one")
    aggregated: dict[str, float] = {}
    for person_id, score_quality_pairs in person_scores.items():
        strongest = sorted(score_quality_pairs, key=lambda pair: pair[0], reverse=True)[:top_k]
        if not strongest:
            continue
        weights = [_quality_weight(quality) for _, quality in strongest]
        weighted_total = sum(score * weight for (score, _), weight in zip(strongest, weights))
        aggregated[person_id] = weighted_total / sum(weights)
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
    sample_quality_by_person: Mapping[str, Sequence[float]] | None = None,
    top_k: int = 2,
) -> MatchDecision:
    """Compare one probe vector with every enrolled sample and decide safely."""

    query = _normalize(query_embedding)
    raw_scores: dict[str, list[float]] = {}
    for person_id, embeddings in embeddings_by_person.items():
        scores = [float(query @ _normalize(embedding)) for embedding in embeddings]
        if scores:
            raw_scores[person_id] = scores
    if sample_quality_by_person is None:
        person_scores = aggregate_person_scores(raw_scores, top_k=top_k)
    else:
        scored_with_quality = {
            person_id: [
                (score, _quality_at(sample_quality_by_person.get(person_id, ()), index))
                for index, score in enumerate(scores)
            ]
            for person_id, scores in raw_scores.items()
        }
        person_scores = aggregate_quality_weighted_scores(scored_with_quality, top_k=top_k)
    return decide_match(
        person_scores,
        match_threshold=match_threshold,
        min_margin=min_margin,
    )


def _stored_quality_score(quality: Mapping[str, float | str]) -> float:
    raw_score = quality["quality_score"]
    if not isinstance(raw_score, (float, int)):
        raise TypeError("stored quality_score must be numeric")
    score = float(raw_score)
    if not 0.0 <= score <= 1.0:
        raise ValueError("stored quality_score must be between 0 and 1")
    return score


def _quality_at(scores: Sequence[float], index: int) -> float:
    return scores[index] if index < len(scores) else 0.6


def _quality_weight(quality_score: float) -> float:
    return 0.5 + 0.5 * max(0.0, min(1.0, quality_score))


def _normalize(embedding: np.ndarray) -> np.ndarray:
    vector = np.asarray(embedding, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if vector.ndim != 1 or vector.size == 0 or norm == 0.0:
        raise ValueError("embedding must be a non-zero one-dimensional vector")
    return vector / norm
