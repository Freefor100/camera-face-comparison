from __future__ import annotations

import bisect
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from camera_face_comparison.open_set_policy import (
    AcceptanceRule,
    neighborhood_aware_cosine,
)


@dataclass(frozen=True)
class JointScoreRow:
    """一个场景中一张有效 Probe 的无阈值候选排序。"""

    scenario: str
    source_identity: str
    expected_person_id: str | None
    top_is_correct: bool
    candidate_scores: tuple[float, ...]

    @property
    def top_score(self) -> float:
        """返回第一候选人员分数。"""

        return self.candidate_scores[0]

    @property
    def score_gap(self) -> float:
        """返回第一、第二候选人员分差。"""

        return self.candidate_scores[0] - self.candidate_scores[1]


@dataclass(frozen=True)
class ScenarioOperatingMetrics:
    """统一工作点在一个图像域场景中的开放集指标。"""

    scenario: str
    protocol_known_total: int
    valid_known_total: int
    valid_unknown_total: int
    rank1_correct: int
    known_true_accepts: int
    known_wrong_accepts: int
    unknown_false_accepts: int
    fpir_valid: float
    tpir_valid: float
    tpir_e2e: float
    fnir_valid: float


@dataclass(frozen=True)
class JointOperatingPoint:
    """一个聚合与接收规则在全部主要场景共享的参数和结果。"""

    aggregation_method: str
    top_k: int
    rule: AcceptanceRule
    target_fpir: float
    minimum_score: float | None
    minimum_gap: float | None
    minimum_probability: float | None
    nac_neighbors: int | None
    meets_all_scenarios: bool
    worst_tpir_e2e: float
    average_tpir_e2e: float
    worst_fpir_valid: float
    scenarios: tuple[ScenarioOperatingMetrics, ...]


def calibrate_joint_rule(
    rows: Sequence[JointScoreRow],
    *,
    aggregation_method: str,
    top_k: int,
    rule: AcceptanceRule,
    target_fpir: float,
    protocol_known_totals: Mapping[str, int],
    nac_neighbors: int | None = None,
) -> JointOperatingPoint:
    """在真实分数断点上为多个场景共同选择一套接收参数。

    参数：
        rows：同一聚合方法在全部主要 Calibration 场景的候选排序。
        aggregation_method、top_k：人员聚合标识。
        rule：四种互斥开放集规则之一。
        target_fpir：每个场景都必须满足的有效 Unknown FPIR 上限。
        protocol_known_totals：各场景质量失败前的 Known 总数。
        nac_neighbors：NAC 使用的邻居数，其他规则不得提供。
    返回：
        优先满足全部场景 FPIR、再最大化最差和平均端到端 TPIR 的工作点。
    """

    if not rows:
        raise ValueError("joint calibration rows must not be empty")
    if not 0.0 <= target_fpir <= 1.0:
        raise ValueError("target_fpir must be between zero and one")
    _validate_rows(rows, protocol_known_totals)
    if rule == "nac":
        if nac_neighbors is None or nac_neighbors < 2:
            raise ValueError("NAC calibration requires at least two neighbors")
    elif nac_neighbors is not None:
        raise ValueError("nac_neighbors is only valid for the NAC rule")

    if rule == "score_threshold":
        values = np.asarray([row.top_score for row in rows], dtype=np.float64)
        selected_value = _select_one_dimensional_candidate(
            rows,
            values=values,
            target_fpir=target_fpir,
            protocol_known_totals=protocol_known_totals,
        )
        parameters = (selected_value, None, None)
    elif rule == "score_gap":
        values = np.asarray([row.score_gap for row in rows], dtype=np.float64)
        selected_value = _select_one_dimensional_candidate(
            rows,
            values=values,
            target_fpir=target_fpir,
            protocol_known_totals=protocol_known_totals,
        )
        parameters = (None, selected_value, None)
    elif rule == "nac":
        probabilities = _nac_probabilities(rows, nac_neighbors or 2)
        selected_value = _select_one_dimensional_candidate(
            rows,
            values=probabilities,
            target_fpir=target_fpir,
            protocol_known_totals=protocol_known_totals,
        )
        parameters = (None, None, selected_value)
    else:
        parameters = _select_combined_candidate(
            rows,
            target_fpir=target_fpir,
            protocol_known_totals=protocol_known_totals,
        )

    return _evaluate_candidate(
        rows,
        aggregation_method=aggregation_method,
        top_k=top_k,
        rule=rule,
        target_fpir=target_fpir,
        protocol_known_totals=protocol_known_totals,
        minimum_score=parameters[0],
        minimum_gap=parameters[1],
        minimum_probability=parameters[2],
        nac_neighbors=nac_neighbors,
    )


def select_joint_operating_point(
    points: Sequence[JointOperatingPoint],
    *,
    minimum_complex_gain: float = 0.01,
) -> JointOperatingPoint:
    """从全部聚合与规则中选择唯一候选，并对复杂规则要求实际收益。

    复杂规则指仅候选分差、双条件和 NAC。若其最差场景端到端 TPIR 相比
    Mean Prototype 最高分阈值不足指定收益，则保留简单基线。
    """

    if not points:
        raise ValueError("joint operating points must not be empty")
    if minimum_complex_gain < 0.0:
        raise ValueError("minimum_complex_gain must not be negative")
    meeting = [point for point in points if point.meets_all_scenarios]
    pool = meeting or list(points)
    selected = min(pool, key=lambda point: _point_key(point, target_met=bool(meeting)))
    baseline = next(
        (
            point
            for point in points
            if point.aggregation_method == "mean_prototype"
            and point.rule == "score_threshold"
            and point.target_fpir == selected.target_fpir
        ),
        None,
    )
    if (
        selected.rule != "score_threshold"
        and baseline is not None
        and baseline.meets_all_scenarios
        and selected.worst_tpir_e2e < baseline.worst_tpir_e2e + minimum_complex_gain
    ):
        return baseline
    return selected


def evaluate_joint_operating_point(
    rows: Sequence[JointScoreRow],
    *,
    selected: JointOperatingPoint,
    protocol_known_totals: Mapping[str, int],
) -> JointOperatingPoint:
    """把 Calibration 冻结的聚合、规则和参数原样应用到 Evaluation。

    本函数不搜索任何断点；返回对象保留原目标与参数，只有各场景指标来自
    新输入行，防止查看 Evaluation 后无意重新选择阈值。
    """

    if not rows:
        raise ValueError("joint Evaluation rows must not be empty")
    _validate_rows(rows, protocol_known_totals)
    return _evaluate_candidate(
        rows,
        aggregation_method=selected.aggregation_method,
        top_k=selected.top_k,
        rule=selected.rule,
        target_fpir=selected.target_fpir,
        protocol_known_totals=protocol_known_totals,
        minimum_score=selected.minimum_score,
        minimum_gap=selected.minimum_gap,
        minimum_probability=selected.minimum_probability,
        nac_neighbors=selected.nac_neighbors,
    )


def accepts_joint_score(row: JointScoreRow, selected: JointOperatingPoint) -> bool:
    """用冻结工作点判断一条候选排序是否被接收，不执行任何重新标定。"""

    return _accepted(
        row,
        rule=selected.rule,
        minimum_score=selected.minimum_score,
        minimum_gap=selected.minimum_gap,
        minimum_probability=selected.minimum_probability,
        nac_neighbors=selected.nac_neighbors,
    )


def _select_one_dimensional_candidate(
    rows: Sequence[JointScoreRow],
    *,
    values: np.ndarray,
    target_fpir: float,
    protocol_known_totals: Mapping[str, int],
) -> float:
    """用排序和后缀累计精确选择一个单变量规则断点。

    每个实际分数只排序一次，随后同时得到全部阈值下的接收数量，避免为
    每个断点重新遍历全部 Probe。返回值仍来自真实分数，不做固定网格近似。
    """

    candidates = np.unique(values)
    true_counts: list[np.ndarray] = []
    unknown_counts: list[np.ndarray] = []
    protocol_totals: list[int] = []
    unknown_totals: list[int] = []
    for scenario in sorted(protocol_known_totals):
        scenario_mask = np.asarray([row.scenario == scenario for row in rows], dtype=bool)
        known_mask = np.asarray(
            [row.expected_person_id is not None for row in rows], dtype=bool
        ) & scenario_mask
        correct_mask = np.asarray([row.top_is_correct for row in rows], dtype=bool) & known_mask
        unknown_mask = scenario_mask & ~known_mask
        scenario_values = values[scenario_mask]
        true_counts.append(
            _suffix_acceptance_counts(
                scenario_values,
                correct_mask[scenario_mask],
                candidates,
            )
        )
        unknown_counts.append(
            _suffix_acceptance_counts(
                scenario_values,
                unknown_mask[scenario_mask],
                candidates,
            )
        )
        protocol_totals.append(protocol_known_totals[scenario])
        unknown_totals.append(int(np.sum(unknown_mask)))

    selected = _select_count_index(
        true_counts=np.stack(true_counts),
        unknown_counts=np.stack(unknown_counts),
        protocol_known_totals=np.asarray(protocol_totals, dtype=np.int64),
        valid_unknown_totals=np.asarray(unknown_totals, dtype=np.int64),
        target_fpir=target_fpir,
    )
    return float(candidates[selected])


def _select_combined_candidate(
    rows: Sequence[JointScoreRow],
    *,
    target_fpir: float,
    protocol_known_totals: Mapping[str, int],
) -> tuple[float, float, None]:
    """精确选择最高分加候选分差规则，不构造二维稠密网格。

    对每个最高分断点，先由各场景 Unknown 的分差顺序统计求出共同满足
    FPIR 所需的最小分差；再用二维离线计数计算 Known 接收数。复杂度约为
    `O((N+C) log N)`，其中 N 为 Probe 数，C 为实际最高分断点数。
    """

    top_scores = np.asarray([row.top_score for row in rows], dtype=np.float64)
    score_gaps = np.asarray([row.score_gap for row in rows], dtype=np.float64)
    candidates = np.unique(top_scores)
    required_gaps = np.zeros(candidates.size, dtype=np.float64)
    true_counts: list[np.ndarray] = []
    unknown_counts: list[np.ndarray] = []
    protocol_totals: list[int] = []
    unknown_totals: list[int] = []

    scenario_masks: dict[str, np.ndarray] = {}
    for scenario in sorted(protocol_known_totals):
        scenario_mask = np.asarray([row.scenario == scenario for row in rows], dtype=bool)
        known_mask = np.asarray(
            [row.expected_person_id is not None for row in rows], dtype=bool
        ) & scenario_mask
        unknown_mask = scenario_mask & ~known_mask
        scenario_masks[scenario] = scenario_mask
        scenario_required = _minimum_gaps_for_unknown_limit(
            top_scores[unknown_mask],
            score_gaps[unknown_mask],
            candidates,
            allowed=math.floor(target_fpir * int(np.sum(unknown_mask)) + 1e-12),
        )
        required_gaps = np.maximum(required_gaps, scenario_required)
        protocol_totals.append(protocol_known_totals[scenario])
        unknown_totals.append(int(np.sum(unknown_mask)))

    legal = required_gaps <= 2.0
    if not np.any(legal):
        # 余弦分差的合法上限仍无法达到目标时，保留边界候选并明确报告未达标。
        required_gaps.fill(2.0)
        legal = np.ones(candidates.size, dtype=bool)
    candidates = candidates[legal]
    required_gaps = required_gaps[legal]

    for scenario in sorted(protocol_known_totals):
        scenario_mask = scenario_masks[scenario]
        selected_rows = [row for row in rows if row.scenario == scenario]
        scenario_scores = top_scores[scenario_mask]
        scenario_gaps = score_gaps[scenario_mask]
        correct = np.asarray(
            [row.expected_person_id is not None and row.top_is_correct for row in selected_rows],
            dtype=bool,
        )
        unknown = np.asarray(
            [row.expected_person_id is None for row in selected_rows], dtype=bool
        )
        true_counts.append(
            _two_dimensional_acceptance_counts(
                scenario_scores,
                scenario_gaps,
                correct,
                candidates,
                required_gaps,
            )
        )
        unknown_counts.append(
            _two_dimensional_acceptance_counts(
                scenario_scores,
                scenario_gaps,
                unknown,
                candidates,
                required_gaps,
            )
        )

    selected = _select_count_index(
        true_counts=np.stack(true_counts),
        unknown_counts=np.stack(unknown_counts),
        protocol_known_totals=np.asarray(protocol_totals, dtype=np.int64),
        valid_unknown_totals=np.asarray(unknown_totals, dtype=np.int64),
        target_fpir=target_fpir,
    )
    return float(candidates[selected]), float(required_gaps[selected]), None


def _suffix_acceptance_counts(
    values: np.ndarray,
    mask: np.ndarray,
    candidates: np.ndarray,
) -> np.ndarray:
    """统计每个候选下 `value >= threshold` 且命中掩码的数量。"""

    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    suffix = np.cumsum(mask[order][::-1], dtype=np.int64)[::-1]
    positions = np.searchsorted(sorted_values, candidates, side="left")
    counts = np.zeros(candidates.size, dtype=np.int64)
    valid = positions < suffix.size
    counts[valid] = suffix[positions[valid]]
    return counts


def _minimum_gaps_for_unknown_limit(
    scores: np.ndarray,
    gaps: np.ndarray,
    thresholds: np.ndarray,
    *,
    allowed: int,
) -> np.ndarray:
    """为每个最高分阈值求最多接收指定 Unknown 数量的最小分差。"""

    order = np.argsort(scores)[::-1]
    query_order = np.argsort(thresholds)[::-1]
    active_gaps: list[float] = []
    required = np.zeros(thresholds.size, dtype=np.float64)
    position = 0
    for query_index in query_order:
        threshold = thresholds[query_index]
        while position < order.size and scores[order[position]] >= threshold:
            bisect.insort(active_gaps, float(gaps[order[position]]))
            position += 1
        if len(active_gaps) > allowed:
            cutoff = active_gaps[-allowed - 1]
            required[query_index] = float(np.nextafter(cutoff, np.inf))
    return required


def _two_dimensional_acceptance_counts(
    scores: np.ndarray,
    gaps: np.ndarray,
    mask: np.ndarray,
    query_scores: np.ndarray,
    query_gaps: np.ndarray,
) -> np.ndarray:
    """用 Fenwick 树统计全部二维 `score >= T and gap >= G` 查询。"""

    coordinates = np.unique(gaps)
    tree = np.zeros(coordinates.size + 1, dtype=np.int64)
    row_order = np.argsort(scores)[::-1]
    query_order = np.argsort(query_scores)[::-1]
    counts = np.zeros(query_scores.size, dtype=np.int64)
    position = 0
    inserted = 0

    for query_index in query_order:
        threshold = query_scores[query_index]
        while position < row_order.size and scores[row_order[position]] >= threshold:
            row_index = row_order[position]
            if mask[row_index]:
                coordinate = int(np.searchsorted(coordinates, gaps[row_index], side="left")) + 1
                _fenwick_add(tree, coordinate)
                inserted += 1
            position += 1
        below = _fenwick_sum(
            tree,
            int(np.searchsorted(coordinates, query_gaps[query_index], side="left")),
        )
        counts[query_index] = inserted - below
    return counts


def _fenwick_add(tree: np.ndarray, index: int) -> None:
    """向 Fenwick 树加入一个离散分差。"""

    while index < tree.size:
        tree[index] += 1
        index += index & -index


def _fenwick_sum(tree: np.ndarray, count: int) -> int:
    """返回 Fenwick 树中前 `count` 个离散坐标的累计数量。"""

    total = 0
    while count > 0:
        total += int(tree[count])
        count -= count & -count
    return total


def _select_count_index(
    *,
    true_counts: np.ndarray,
    unknown_counts: np.ndarray,
    protocol_known_totals: np.ndarray,
    valid_unknown_totals: np.ndarray,
    target_fpir: float,
) -> int:
    """按共同 FPIR、最差端到端 TPIR 和平均 TPIR 选择稳定索引。"""

    fpirs = unknown_counts / valid_unknown_totals[:, None]
    tpirs = true_counts / protocol_known_totals[:, None]
    worst_fpir = np.max(fpirs, axis=0)
    worst_tpir = np.min(tpirs, axis=0)
    average_tpir = np.mean(tpirs, axis=0)
    meets = np.all(fpirs <= target_fpir + 1e-12, axis=0)
    pool = np.flatnonzero(meets)
    if pool.size:
        return min(pool, key=lambda index: (-worst_tpir[index], -average_tpir[index]))
    return min(
        range(worst_fpir.size),
        key=lambda index: (worst_fpir[index], -worst_tpir[index], -average_tpir[index]),
    )


def _evaluate_candidate(
    rows: Sequence[JointScoreRow],
    *,
    aggregation_method: str,
    top_k: int,
    rule: AcceptanceRule,
    target_fpir: float,
    protocol_known_totals: Mapping[str, int],
    minimum_score: float | None,
    minimum_gap: float | None,
    minimum_probability: float | None,
    nac_neighbors: int | None,
) -> JointOperatingPoint:
    """把一组明确参数应用到所有场景并生成可比较指标。"""

    metrics: list[ScenarioOperatingMetrics] = []
    for scenario in sorted(protocol_known_totals):
        selected_rows = [row for row in rows if row.scenario == scenario]
        accepted = [
            _accepted(
                row,
                rule=rule,
                minimum_score=minimum_score,
                minimum_gap=minimum_gap,
                minimum_probability=minimum_probability,
                nac_neighbors=nac_neighbors,
            )
            for row in selected_rows
        ]
        known = [row.expected_person_id is not None for row in selected_rows]
        correct = [row.top_is_correct and is_known for row, is_known in zip(selected_rows, known)]
        valid_known_total = sum(known)
        valid_unknown_total = len(known) - valid_known_total
        if valid_known_total == 0 or valid_unknown_total == 0:
            raise ValueError(f"scenario requires valid Known and Unknown rows: {scenario}")
        true_accepts = sum(a and c for a, c in zip(accepted, correct))
        wrong_accepts = sum(
            a and is_known and not c for a, is_known, c in zip(accepted, known, correct)
        )
        unknown_accepts = sum(a and not is_known for a, is_known in zip(accepted, known))
        protocol_known_total = protocol_known_totals[scenario]
        metrics.append(
            ScenarioOperatingMetrics(
                scenario=scenario,
                protocol_known_total=protocol_known_total,
                valid_known_total=valid_known_total,
                valid_unknown_total=valid_unknown_total,
                rank1_correct=sum(correct),
                known_true_accepts=true_accepts,
                known_wrong_accepts=wrong_accepts,
                unknown_false_accepts=unknown_accepts,
                fpir_valid=unknown_accepts / valid_unknown_total,
                tpir_valid=true_accepts / valid_known_total,
                tpir_e2e=true_accepts / protocol_known_total,
                fnir_valid=1.0 - true_accepts / valid_known_total,
            )
        )
    tpirs = [metric.tpir_e2e for metric in metrics]
    fpirs = [metric.fpir_valid for metric in metrics]
    return JointOperatingPoint(
        aggregation_method=aggregation_method,
        top_k=top_k,
        rule=rule,
        target_fpir=target_fpir,
        minimum_score=minimum_score,
        minimum_gap=minimum_gap,
        minimum_probability=minimum_probability,
        nac_neighbors=nac_neighbors,
        meets_all_scenarios=all(value <= target_fpir + 1e-12 for value in fpirs),
        worst_tpir_e2e=min(tpirs),
        average_tpir_e2e=float(np.mean(tpirs)),
        worst_fpir_valid=max(fpirs),
        scenarios=tuple(metrics),
    )


def _accepted(
    row: JointScoreRow,
    *,
    rule: AcceptanceRule,
    minimum_score: float | None,
    minimum_gap: float | None,
    minimum_probability: float | None,
    nac_neighbors: int | None,
) -> bool:
    """执行一个候选参数组合，不借助哨兵值关闭条件。"""

    if rule == "score_threshold":
        return row.top_score >= _required(minimum_score)
    if rule == "score_gap":
        return row.score_gap >= _required(minimum_gap)
    if rule == "score_and_gap":
        return row.top_score >= _required(minimum_score) and row.score_gap >= _required(
            minimum_gap
        )
    probability = neighborhood_aware_cosine(
        list(row.candidate_scores), neighbor_count=nac_neighbors or 0
    )
    return probability >= _required(minimum_probability)


def _nac_probabilities(rows: Sequence[JointScoreRow], neighbor_count: int) -> np.ndarray:
    """批量计算 NAC 第一候选概率，供真实断点扫描复用。"""

    return np.asarray(
        [
            neighborhood_aware_cosine(list(row.candidate_scores), neighbor_count=neighbor_count)
            for row in rows
        ],
        dtype=np.float64,
    )


def _point_key(
    point: JointOperatingPoint,
    *,
    target_met: bool,
) -> tuple[float | int | str, ...]:
    """生成满足 FPIR、最差 TPIR、平均 TPIR和复杂度顺序的稳定选择键。"""

    method_cost = {
        "mean_prototype": 0,
        "single": 1,
        "max": 2,
        "top_k_mean": 3 + point.top_k,
    }.get(point.aggregation_method, 100)
    rule_cost = {"score_threshold": 0, "score_gap": 1, "score_and_gap": 2, "nac": 3}[
        point.rule
    ]
    return (
        0.0 if target_met else point.worst_fpir_valid,
        -point.worst_tpir_e2e,
        -point.average_tpir_e2e,
        rule_cost,
        method_cost,
        point.aggregation_method,
        point.top_k,
    )


def _validate_rows(
    rows: Sequence[JointScoreRow],
    protocol_known_totals: Mapping[str, int],
) -> None:
    """校验场景覆盖、候选排序和端到端分母。"""

    scenarios = {row.scenario for row in rows}
    if scenarios != set(protocol_known_totals):
        raise ValueError("protocol Known totals must match all calibration scenarios")
    for row in rows:
        if len(row.candidate_scores) < 2:
            raise ValueError("joint calibration requires at least two candidate scores")
        values = np.asarray(row.candidate_scores, dtype=np.float64)
        if np.any(~np.isfinite(values)) or np.any(values[:-1] < values[1:]):
            raise ValueError("candidate scores must be finite and sorted descending")
        if row.expected_person_id is None and row.top_is_correct:
            raise ValueError("Unknown Probe cannot have a correct Gallery identity")
    for scenario, total in protocol_known_totals.items():
        valid_known = sum(
            row.scenario == scenario and row.expected_person_id is not None for row in rows
        )
        if total < valid_known:
            raise ValueError("protocol Known total cannot be smaller than valid Known rows")


def _required(value: float | None) -> float:
    """读取当前规则必需参数，缺失时立即报告模型错误。"""

    if value is None:
        raise ValueError("required policy parameter is missing")
    return value
