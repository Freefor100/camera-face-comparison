from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from camera_face_comparison.calibration import CalibrationRecord, calibrate_thresholds
from camera_face_comparison.config import load_settings, write_quality_tier_thresholds


def _values(start: int, stop: int) -> list[float]:
    """生成指定百分比范围内、步长为百分之一的阈值候选。"""
    return [value / 100 for value in range(start, stop + 1)]


def _load_records(path: Path) -> list[CalibrationRecord]:
    """读取 JSONL 标定记录并转换为领域对象。

    参数：
        path：每行包含真实身份和人级别得分的 JSONL 文件。
    返回：
        可供阈值搜索使用的标定记录列表。
    前置条件：
        每行必须包含 `expected_person_id` 和 `person_scores` 字段。
    """
    records: list[CalibrationRecord] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            expected = payload["expected_person_id"]
            scores = {str(key): float(value) for key, value in payload["person_scores"].items()}
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid calibration record at line {line_number}") from error
        if expected is not None:
            expected = str(expected)
        records.append(CalibrationRecord(expected_person_id=expected, person_scores=scores))
    return records


def main() -> int:
    """读取标定数据、搜索阈值并写回指定质量层级的配置。"""
    parser = argparse.ArgumentParser(
        description="Calibrate the local match threshold and candidate-gap threshold."
    )
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument(
        "--scores",
        type=Path,
        default=PROJECT_ROOT / "data" / "calibration_scores.jsonl",
        help="JSONL rows: expected_person_id (string/null), person_scores (object)",
    )
    parser.add_argument(
        "--quality-tier",
        choices=("high", "medium"),
        default="high",
        help="probe-quality policy to calibrate; run once per held-out quality subset",
    )
    args = parser.parse_args()
    if not args.scores.is_file():
        print(f"Calibration input does not exist: {args.scores}", file=sys.stderr)
        return 1
    records = _load_records(args.scores)
    result = calibrate_thresholds(
        records=records,
        threshold_candidates=_values(30, 80),
        margin_candidates=_values(0, 20),
    )
    settings = load_settings(args.data_dir)
    write_quality_tier_thresholds(
        settings,
        tier=args.quality_tier,
        match_threshold=result.match_threshold,
        min_margin=result.min_margin,
    )
    print(
        json.dumps(
            {
                "match_threshold": result.match_threshold,
                "min_margin": result.min_margin,
                "quality_tier": args.quality_tier,
                "unknown_false_accepts": result.unknown_false_accepts,
                "known_correct": result.known_correct,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
