from __future__ import annotations

import math

import pytest

from camera_face_comparison.open_set_policy import (
    NacPolicy,
    ScoreAndGapPolicy,
    ScoreGapPolicy,
    ScoreThresholdPolicy,
    apply_open_set_policy,
)


def test_score_threshold_accepts_only_when_the_best_identity_reaches_the_threshold() -> None:
    """最高分规则不能因为候选分差很大而绕过最低相似度。"""

    scores = {"alice": 0.80, "bob": 0.10}

    accepted = apply_open_set_policy(scores, ScoreThresholdPolicy(minimum_score=0.75))
    rejected = apply_open_set_policy(scores, ScoreThresholdPolicy(minimum_score=0.85))

    assert accepted.accepted_person_id == "alice"
    assert accepted.acceptance_score == pytest.approx(0.80)
    assert accepted.rule == "score_threshold"
    assert rejected.accepted_person_id is None
    assert rejected.reason == "score_below_threshold"


def test_score_gap_rule_does_not_apply_an_implicit_score_threshold() -> None:
    """仅候选分差规则必须接受低分但分离明显的候选，避免重现负一哨兵。"""

    decision = apply_open_set_policy(
        {"alice": 0.20, "bob": -0.20},
        ScoreGapPolicy(minimum_gap=0.35),
    )

    assert decision.accepted_person_id == "alice"
    assert decision.score_gap == pytest.approx(0.40)
    assert decision.acceptance_score == pytest.approx(0.40)
    assert decision.rule == "score_gap"


def test_score_and_gap_requires_both_conditions() -> None:
    """联合规则缺少任一条件时都必须拒绝，不能退化为单变量规则。"""

    policy = ScoreAndGapPolicy(minimum_score=0.70, minimum_gap=0.20)

    assert apply_open_set_policy({"alice": 0.75, "bob": 0.50}, policy).accepted
    assert (
        apply_open_set_policy({"alice": 0.65, "bob": 0.20}, policy).reason
        == "score_below_threshold"
    )
    assert (
        apply_open_set_policy({"alice": 0.75, "bob": 0.65}, policy).reason
        == "score_gap_below_minimum"
    )


def test_nac_uses_the_local_softmax_probability_as_acceptance_score() -> None:
    """NAC 必须按前 K 个身份的 softmax 占比计算，而不是复用候选分差。"""

    decision = apply_open_set_policy(
        {"alice": 1.0, "bob": 0.0, "carol": -1.0},
        NacPolicy(neighbor_count=2, minimum_probability=0.70),
    )

    expected = math.e / (math.e + 1.0)
    assert decision.accepted_person_id == "alice"
    assert decision.acceptance_score == pytest.approx(expected)
    assert decision.rule == "nac"


def test_relative_rules_reject_when_the_gallery_has_only_one_identity() -> None:
    """仅一人标准库不能伪造候选分差或得到恒为一的 NAC 置信度。"""

    for policy in (
        ScoreGapPolicy(minimum_gap=0.1),
        ScoreAndGapPolicy(minimum_score=0.1, minimum_gap=0.1),
        NacPolicy(neighbor_count=4, minimum_probability=0.5),
    ):
        decision = apply_open_set_policy({"alice": 0.9}, policy)
        assert not decision.accepted
        assert decision.reason == "insufficient_gallery_identities"


def test_nac_uses_all_available_identities_when_k_exceeds_gallery_size() -> None:
    """NAC 的 K 大于身份数时必须使用全部身份，且仍保持有限概率。"""

    decision = apply_open_set_policy(
        {"alice": 0.9, "bob": 0.5, "carol": 0.1},
        NacPolicy(neighbor_count=16, minimum_probability=0.0),
    )

    expected = math.exp(0.9) / (math.exp(0.9) + math.exp(0.5) + math.exp(0.1))
    assert decision.acceptance_score == pytest.approx(expected)

