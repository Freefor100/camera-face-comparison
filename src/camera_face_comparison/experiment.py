from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .config import QualityTierPolicy
from .recognition import (
    MatchDecision,
    aggregate_person_scores,
    aggregate_quality_weighted_scores,
    decide_match,
)


@dataclass(frozen=True)
class ExperimentRecord:
    """Scores for one labelled probe image, kept independently of private images."""

    expected_person_id: str | None
    sample_scores: Mapping[str, Sequence[float]]
    latency_ms: float
    sample_quality_scores: Mapping[str, Sequence[float]] | None = None
    probe_quality_tier: str | None = None


@dataclass(frozen=True)
class ExperimentMetrics:
    total: int
    known_total: int
    known_correct: int
    unknown_total: int
    unknown_rejected: int
    misidentifications: int
    average_latency_ms: float

    @property
    def known_accuracy(self) -> float | None:
        return self.known_correct / self.known_total if self.known_total else None

    @property
    def unknown_rejection_rate(self) -> float | None:
        return self.unknown_rejected / self.unknown_total if self.unknown_total else None

    @property
    def false_positive_identification_rate(self) -> float | None:
        """Unknown probes incorrectly assigned to a library identity (FPIR)."""

        if not self.unknown_total:
            return None
        return (self.unknown_total - self.unknown_rejected) / self.unknown_total

    @property
    def false_negative_identification_rate(self) -> float | None:
        """Known probes that were rejected or assigned to the wrong identity (FNIR)."""

        if not self.known_total:
            return None
        return (self.known_total - self.known_correct) / self.known_total

    @property
    def rank_one_identification_rate(self) -> float | None:
        """Rank-1 closed-set-style correctness reported alongside the open-set rates."""

        return self.known_accuracy


def evaluate_experiments(
    *,
    records: Sequence[ExperimentRecord],
    match_threshold: float,
    min_margin: float,
    top_k: int = 2,
    quality_tiers: Mapping[str, QualityTierPolicy] | None = None,
) -> tuple[ExperimentMetrics, ExperimentMetrics]:
    """Evaluate the simple baseline and the robust rule on exactly the same probes."""

    baseline_decisions = [
        decide_match(
            _highest_sample_scores(record.sample_scores),
            match_threshold=match_threshold,
            min_margin=0.0,
        )
        for record in records
    ]
    optimized_decisions = [
        decide_match(
            _optimized_person_scores(record, top_k=top_k),
            match_threshold=_policy_for(record, quality_tiers, match_threshold, min_margin).match_threshold,
            min_margin=_policy_for(record, quality_tiers, match_threshold, min_margin).min_margin,
        )
        for record in records
    ]
    return (
        _metrics(records, baseline_decisions),
        _metrics(records, optimized_decisions),
    )


def _optimized_person_scores(record: ExperimentRecord, *, top_k: int) -> dict[str, float]:
    if record.sample_quality_scores is None:
        return aggregate_person_scores(record.sample_scores, top_k=top_k)
    scores_with_quality = {
        person_id: [
            (score, _quality_at(record.sample_quality_scores.get(person_id, ()), index))
            for index, score in enumerate(sample_scores)
        ]
        for person_id, sample_scores in record.sample_scores.items()
    }
    return aggregate_quality_weighted_scores(scores_with_quality, top_k=top_k)


def _policy_for(
    record: ExperimentRecord,
    quality_tiers: Mapping[str, QualityTierPolicy] | None,
    match_threshold: float,
    min_margin: float,
) -> QualityTierPolicy:
    if quality_tiers is not None and record.probe_quality_tier in quality_tiers:
        return quality_tiers[record.probe_quality_tier]
    return QualityTierPolicy(match_threshold=match_threshold, min_margin=min_margin)


def _quality_at(scores: Sequence[float], index: int) -> float:
    return scores[index] if index < len(scores) else 0.6


def _highest_sample_scores(
    sample_scores: Mapping[str, Sequence[float]],
) -> dict[str, float]:
    return {
        person_id: max(scores)
        for person_id, scores in sample_scores.items()
        if scores
    }


def _metrics(
    records: Sequence[ExperimentRecord], decisions: Sequence[MatchDecision]
) -> ExperimentMetrics:
    known_total = 0
    known_correct = 0
    unknown_total = 0
    unknown_rejected = 0
    misidentifications = 0

    for record, decision in zip(records, decisions, strict=True):
        if record.expected_person_id is None:
            unknown_total += 1
            if decision.status != "matched":
                unknown_rejected += 1
            else:
                misidentifications += 1
            continue
        known_total += 1
        if decision.status == "matched" and decision.person_id == record.expected_person_id:
            known_correct += 1
        elif decision.status == "matched":
            misidentifications += 1

    total = len(records)
    average_latency_ms = sum(record.latency_ms for record in records) / total if total else 0.0
    return ExperimentMetrics(
        total=total,
        known_total=known_total,
        known_correct=known_correct,
        unknown_total=unknown_total,
        unknown_rejected=unknown_rejected,
        misidentifications=misidentifications,
        average_latency_ms=average_latency_ms,
    )
