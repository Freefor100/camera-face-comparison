from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from .recognition import decide_match


@dataclass(frozen=True)
class CalibrationRecord:
    """One held-out probe represented by already-computed person-level scores."""

    expected_person_id: str | None
    person_scores: Mapping[str, float]


@dataclass(frozen=True)
class CalibrationResult:
    """The safest tested threshold pair and its measured outcomes."""

    match_threshold: float
    min_margin: float
    unknown_false_accepts: int
    known_correct: int


def calibrate_thresholds(
    *,
    records: Sequence[CalibrationRecord],
    threshold_candidates: Iterable[float],
    margin_candidates: Iterable[float],
) -> CalibrationResult:
    """Select thresholds by rejecting unknown identities before maximizing known matches."""

    thresholds = sorted({float(value) for value in threshold_candidates})
    margins = sorted({float(value) for value in margin_candidates})
    if not records or not thresholds or not margins:
        raise ValueError("records, threshold_candidates, and margin_candidates must not be empty")

    candidates: list[CalibrationResult] = []
    for threshold in thresholds:
        for margin in margins:
            unknown_false_accepts = 0
            known_correct = 0
            for record in records:
                decision = decide_match(
                    record.person_scores,
                    match_threshold=threshold,
                    min_margin=margin,
                )
                if record.expected_person_id is None:
                    unknown_false_accepts += int(decision.status == "matched")
                else:
                    known_correct += int(decision.person_id == record.expected_person_id)
            candidates.append(
                CalibrationResult(
                    match_threshold=threshold,
                    min_margin=margin,
                    unknown_false_accepts=unknown_false_accepts,
                    known_correct=known_correct,
                )
            )

    return max(
        candidates,
        key=lambda item: (
            -item.unknown_false_accepts,
            item.known_correct,
            item.match_threshold,
            item.min_margin,
        ),
    )
