from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .recognition import MatchDecision, aggregate_person_scores, decide_match


@dataclass(frozen=True)
class ExperimentRecord:
    """Scores for one labelled probe image, kept independently of private images."""

    expected_person_id: str | None
    sample_scores: Mapping[str, Sequence[float]]
    latency_ms: float


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


def evaluate_experiments(
    *,
    records: Sequence[ExperimentRecord],
    match_threshold: float,
    min_margin: float,
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
            aggregate_person_scores(record.sample_scores),
            match_threshold=match_threshold,
            min_margin=min_margin,
        )
        for record in records
    ]
    return (
        _metrics(records, baseline_decisions),
        _metrics(records, optimized_decisions),
    )


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
