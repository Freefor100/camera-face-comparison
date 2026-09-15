from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from experiments.face_evaluation.experiment_artifacts import file_sha256, write_json_atomic


def main() -> int:
    """在两边都有效的同一批官方 Pair 上比较原始 LFW 与 XQLFW。"""

    parser = argparse.ArgumentParser(
        description="Compare original-LFW and XQLFW reports on common valid pairs."
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "experiments"
        / "phase5"
        / "lfw_xqlfw_pairs_baseline_report.json",
    )
    parser.add_argument(
        "--cross-quality",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "experiments"
        / "phase5"
        / "xqlfw_evaluation_report.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "experiments"
        / "phase5"
        / "xqlfw_domain_comparison.json",
    )
    args = parser.parse_args()
    try:
        baseline = _read_report(args.baseline)
        cross_quality = _read_report(args.cross_quality)
        payload = compare_reports(baseline, cross_quality) | {
            "artifact": "xqlfw-common-pair-domain-comparison-v1",
            "generated_at": datetime.now(UTC).isoformat(),
            "baseline_report": str(args.baseline),
            "baseline_report_sha256": file_sha256(args.baseline),
            "cross_quality_report": str(args.cross_quality),
            "cross_quality_report_sha256": file_sha256(args.cross_quality),
        }
        write_json_atomic(args.output, payload)
    except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1
    print(json.dumps(payload["overall"], ensure_ascii=False))
    return 0


def compare_reports(baseline: dict, cross_quality: dict) -> dict:
    """按路径和折次连接两个报告，并量化共同有效 Pair 的变化。"""

    baseline_result = baseline["result"]
    cross_result = cross_quality["result"]
    baseline_pairs = {_pair_key(pair): pair for pair in baseline_result["pairs"]}
    cross_pairs = {_pair_key(pair): pair for pair in cross_result["pairs"]}
    common_keys = sorted(set(baseline_pairs) & set(cross_pairs))
    if not common_keys:
        raise ValueError("XQLFW reports have no common valid pairs")
    joined = [(baseline_pairs[key], cross_pairs[key]) for key in common_keys]
    baseline_correct = sum(bool(left["correct"]) for left, _ in joined)
    cross_correct = sum(bool(right["correct"]) for _, right in joined)
    common_total = len(joined)
    overall = {
        "baseline_valid_pair_total": int(baseline_result["valid_pair_total"]),
        "cross_quality_valid_pair_total": int(cross_result["valid_pair_total"]),
        "common_valid_pair_total": common_total,
        "baseline_accuracy": baseline_correct / common_total,
        "cross_quality_accuracy": cross_correct / common_total,
        "accuracy_delta": (cross_correct - baseline_correct) / common_total,
        "baseline_valid_image_total": int(baseline_result["valid_image_total"]),
        "cross_quality_valid_image_total": int(cross_result["valid_image_total"]),
        "baseline_fte_total": len(baseline_result["rejected_images"]),
        "cross_quality_fte_total": len(cross_result["rejected_images"]),
    }
    transitions = {
        "both_correct": sum(left["correct"] and right["correct"] for left, right in joined),
        "baseline_correct_cross_quality_wrong": sum(
            left["correct"] and not right["correct"] for left, right in joined
        ),
        "baseline_wrong_cross_quality_correct": sum(
            not left["correct"] and right["correct"] for left, right in joined
        ),
        "both_wrong": sum(
            not left["correct"] and not right["correct"] for left, right in joined
        ),
    }
    return {
        "overall": overall,
        "transitions": transitions,
        "positive_similarity": _similarity_comparison(
            [pair for pair in joined if pair[1]["same_identity"]]
        ),
        "negative_similarity": _similarity_comparison(
            [pair for pair in joined if not pair[1]["same_identity"]]
        ),
        "min_quality_bins": _comparison_bins(
            joined,
            value=lambda pair: float(pair[1]["min_quality"]),
            boundaries=(0.0, 0.4, 0.6, 0.8, 1.0),
        ),
        "quality_gap_bins": _comparison_bins(
            joined,
            value=lambda pair: float(pair[1]["quality_gap"]),
            boundaries=(0.0, 0.1, 0.2, 0.3, 0.5, 1.0),
        ),
    }


def _comparison_bins(
    pairs: Sequence[tuple[dict, dict]],
    *,
    value: Callable[[tuple[dict, dict]], float],
    boundaries: Sequence[float],
) -> list[dict[str, float | int | str | None]]:
    """按固定质量边界比较两个域在共同 Pair 上的准确率。"""

    bins: list[dict[str, float | int | str | None]] = []
    for index in range(len(boundaries) - 1):
        lower = boundaries[index]
        upper = boundaries[index + 1]
        is_last = index == len(boundaries) - 2
        if is_last:
            selected = [pair for pair in pairs if lower <= value(pair) <= upper]
        else:
            selected = [pair for pair in pairs if lower <= value(pair) < upper]
        closing = "]" if is_last else ")"
        if selected:
            baseline_accuracy = sum(pair[0]["correct"] for pair in selected) / len(selected)
            cross_accuracy = sum(pair[1]["correct"] for pair in selected) / len(selected)
        else:
            baseline_accuracy = cross_accuracy = None
        bins.append(
            {
                "label": f"[{lower:.2f}, {upper:.2f}{closing}",
                "pair_total": len(selected),
                "baseline_accuracy": baseline_accuracy,
                "cross_quality_accuracy": cross_accuracy,
                "accuracy_delta": (
                    cross_accuracy - baseline_accuracy
                    if baseline_accuracy is not None and cross_accuracy is not None
                    else None
                ),
            }
        )
    return bins


def _similarity_comparison(pairs: Sequence[tuple[dict, dict]]) -> dict[str, float]:
    """比较共同 Pair 在两个图像域中的相似度均值和中位数。"""

    baseline = [float(pair[0]["similarity"]) for pair in pairs]
    cross_quality = [float(pair[1]["similarity"]) for pair in pairs]
    return {
        "pair_total": len(pairs),
        "baseline_mean": statistics.fmean(baseline),
        "cross_quality_mean": statistics.fmean(cross_quality),
        "mean_delta": statistics.fmean(cross_quality) - statistics.fmean(baseline),
        "baseline_median": statistics.median(baseline),
        "cross_quality_median": statistics.median(cross_quality),
        "median_delta": statistics.median(cross_quality) - statistics.median(baseline),
    }


def _pair_key(pair: dict) -> tuple[int, str, str]:
    """生成不依赖阈值和预测结果的官方 Pair 连接键。"""

    return int(pair["fold"]), str(pair["left_path"]), str(pair["right_path"])


def _read_report(path: Path) -> dict:
    """读取并检查 XQLFW JSON 报告的基础结构。"""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("result"), dict):
        raise TypeError(f"invalid XQLFW report: {path}")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
