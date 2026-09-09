from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from camera_face_comparison.config import Settings, load_settings
from camera_face_comparison.evaluation_cache import (
    embedding_extraction_id,
    file_sha256,
    quality_policy_id,
    write_json_atomic,
)
from camera_face_comparison.quality_analysis import (
    QualityUtilityRecord,
    analyze_gallery_quality,
    analyze_probe_quality,
    error_vs_reject,
    known_rank1_outcomes,
    load_quality_measurements,
    quality_error_bins,
    quality_metric_distributions,
    quality_utility_values,
)
from camera_face_comparison.quality_experiment import read_quality_experiment_protocol
from camera_face_comparison.quality_experiment_store import QualityExperimentEntry


def main() -> int:
    """从 Phase 3 原始 SQLite 生成不重新调用模型的质量证据报告。"""

    parser = argparse.ArgumentParser(
        description="Analyze the completed Phase 3 quality experiment without inference."
    )
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--store", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--min-face-size",
        type=int,
        default=80,
        help="必须与 Phase 3 LFW 实验的质量策略一致",
    )
    args = parser.parse_args()
    if args.min_face_size < 1:
        parser.error("--min-face-size must be at least one")

    settings = replace(load_settings(args.data_dir), min_face_size_px=args.min_face_size)
    phase3_dir = settings.data_dir / "experiments" / "phase3"
    protocol_path = args.protocol or phase3_dir / "protocol.json"
    manifest_path = args.manifest or phase3_dir / "manifest.json"
    store_path = args.store or phase3_dir / "measurements.sqlite"
    output_path = args.output or phase3_dir / "report.json"
    try:
        report = build_report(
            settings=settings,
            protocol_path=protocol_path,
            manifest_path=manifest_path,
            store_path=store_path,
        )
        write_json_atomic(output_path, report)
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1
    print(f"quality report written: {output_path}")
    return 0


def build_report(
    *,
    settings: Settings,
    protocol_path: Path,
    manifest_path: Path,
    store_path: Path,
) -> dict[str, object]:
    """验证实验产物并组装 Probe、Gallery 和质量拒绝统计。"""

    protocol = read_quality_experiment_protocol(protocol_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["artifact"] != "lfw-quality-experiment-v1":
        raise ValueError("unsupported quality experiment manifest")
    if manifest["status"] != "completed":
        raise RuntimeError("quality experiment is not completed")
    if manifest["protocol_sha256"] != file_sha256(protocol_path):
        raise RuntimeError("quality experiment protocol hash does not match manifest")
    extraction_id = embedding_extraction_id(settings)
    policy_id = quality_policy_id(settings)
    if manifest["embedding_extraction_id"] != extraction_id:
        raise RuntimeError("embedding extraction configuration does not match experiment")
    if manifest["quality_policy_id"] != policy_id:
        raise RuntimeError("quality policy does not match experiment")

    measurements = load_quality_measurements(store_path, extraction_id)
    gallery_paths = {
        str(person_id): str(path)
        for person_id, path in manifest["summary"]["gallery_references"].items()
    }
    if set(gallery_paths) != {case.person_id for case in protocol.known}:
        raise RuntimeError("manifest Gallery references do not match quality protocol")
    conditions = tuple(dict(item) for item in manifest["conditions"])
    condition_keys = tuple(str(item["key"]) for item in conditions)
    baseline_key = "baseline:1"
    if baseline_key not in measurements:
        raise RuntimeError("baseline measurements are missing")

    known_paths = {case.person_id: case.probe_path for case in protocol.known}
    unknown_paths = tuple(case.probe_path for case in protocol.unknown)
    baseline_gallery = _entries_by_identity(
        measurements[baseline_key], gallery_paths, baseline_key
    )
    baseline_known = _entries_by_identity(
        measurements[baseline_key], known_paths, baseline_key
    )
    baseline_unknown = _entries_by_path(
        measurements[baseline_key], unknown_paths, baseline_key
    )

    probe_conditions: list[dict[str, object]] = []
    gallery_conditions: list[dict[str, object]] = []
    probe_utility_records: dict[str, list[QualityUtilityRecord]] = {}
    gallery_utility_records: dict[str, list[QualityUtilityRecord]] = {}
    for condition in conditions:
        key = str(condition["key"])
        condition_entries = measurements.get(key)
        if condition_entries is None:
            raise RuntimeError(f"quality condition is missing: {key}")
        gallery = _entries_by_identity(condition_entries, gallery_paths, key)
        known = _entries_by_identity(condition_entries, known_paths, key)
        unknown = _entries_by_path(condition_entries, unknown_paths, key)

        probe_summary = analyze_probe_quality(
            gallery_entries=baseline_gallery,
            known_entries=known,
            unknown_entries=unknown,
            settings=settings,
        )
        gallery_summary = analyze_gallery_quality(
            gallery_entries=gallery,
            known_entries=baseline_known,
            unknown_entries=baseline_unknown,
            settings=settings,
        )
        probe_conditions.append(
            condition
            | {
                "summary": asdict(probe_summary),
                "quality_metrics": {
                    "known": _distribution_dict(known.values(), settings),
                    "unknown": _distribution_dict(unknown, settings),
                },
            }
        )
        gallery_conditions.append(
            condition
            | {
                "summary": asdict(gallery_summary),
                "quality_metrics": _distribution_dict(gallery.values(), settings),
            }
        )
        _append_utility_records(
            probe_utility_records,
            entries=known,
            outcomes=known_rank1_outcomes(
                gallery_entries=baseline_gallery,
                known_entries=known,
            ),
            settings=settings,
        )
        _append_utility_records(
            gallery_utility_records,
            entries=gallery,
            outcomes=known_rank1_outcomes(
                gallery_entries=gallery,
                known_entries=baseline_known,
            ),
            settings=settings,
        )

    clean_probe_values = _utility_values_by_name(baseline_known.values(), settings)
    clean_gallery_values = _utility_values_by_name(baseline_gallery.values(), settings)
    return {
        "artifact": "lfw-quality-analysis-v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "protocol_sha256": file_sha256(protocol_path),
        "source_manifest_sha256": file_sha256(manifest_path),
        "embedding_extraction_id": extraction_id,
        "quality_policy_id": policy_id,
        "execution_backend": manifest["execution_backend"],
        "quality_configuration": _quality_configuration(settings),
        "coverage": {
            "known_identity_total": len(protocol.known),
            "unknown_identity_total": len(protocol.unknown),
            "condition_total": len(condition_keys),
            "official_source_total": len(gallery_paths) + len(known_paths) + len(unknown_paths),
            "official_measurement_total": (
                len(gallery_paths) + len(known_paths) + len(unknown_paths)
            )
            * len(condition_keys),
        },
        "interpretation": {
            "probe_conditions": "退化 Probe，Gallery 固定为 baseline",
            "gallery_conditions": "退化 Gallery，Probe 固定为 baseline",
            "rank1_denominator": "协议中的全部 116 个 Known 身份，FTE 和质量拒绝不从分母删除",
            "unknown_scores": "连续最高候选分数；Phase 3 未冻结匹配阈值，因此不报告 FPIR",
            "accepted_rank1": "应用当前质量门后仍被接收且 Rank-1 正确；不是开放集 TPIR",
            "error_vs_reject": "阈值由 baseline 干净输入的目标保留率确定，再应用到全部单因素条件",
        },
        "probe_conditions": probe_conditions,
        "gallery_conditions": gallery_conditions,
        "error_vs_reject": {
            "probe": _error_vs_reject_report(probe_utility_records, clean_probe_values),
            "gallery": _error_vs_reject_report(gallery_utility_records, clean_gallery_values),
        },
    }


def _entries_by_identity(
    condition_entries: dict[str, QualityExperimentEntry],
    identity_paths: dict[str, str],
    condition_key: str,
) -> dict[str, QualityExperimentEntry]:
    """按固定身份到路径映射取出一个条件的记录，缺失时拒绝生成报告。"""

    result: dict[str, QualityExperimentEntry] = {}
    for identity, relative_path in sorted(identity_paths.items()):
        if relative_path not in condition_entries:
            raise RuntimeError(f"missing measurement: {condition_key} {relative_path}")
        result[identity] = condition_entries[relative_path]
    return result


def _entries_by_path(
    condition_entries: dict[str, QualityExperimentEntry],
    paths: tuple[str, ...],
    condition_key: str,
) -> tuple[QualityExperimentEntry, ...]:
    """按协议顺序取出 Unknown 记录，缺失时拒绝生成报告。"""

    result: list[QualityExperimentEntry] = []
    for relative_path in paths:
        if relative_path not in condition_entries:
            raise RuntimeError(f"missing measurement: {condition_key} {relative_path}")
        result.append(condition_entries[relative_path])
    return tuple(result)


def _distribution_dict(
    entries,
    settings: Settings,
) -> dict[str, dict[str, object]]:
    """把质量指标分布 dataclass 转换为 JSON 字典。"""

    return {
        name: asdict(distribution)
        for name, distribution in quality_metric_distributions(entries, settings).items()
    }


def _append_utility_records(
    target: dict[str, list[QualityUtilityRecord]],
    *,
    entries: dict[str, QualityExperimentEntry],
    outcomes: dict[str, bool | None],
    settings: Settings,
) -> None:
    """把一个条件下可归因的 Known 结果追加到各质量效用序列。"""

    for identity, entry in entries.items():
        outcome = outcomes[identity]
        if outcome is None or entry.status != "observed":
            continue
        for name, value in quality_utility_values(entry, settings).items():
            if name == "brightness":
                continue
            target.setdefault(name, []).append(QualityUtilityRecord(value, outcome))


def _utility_values_by_name(
    entries,
    settings: Settings,
) -> dict[str, tuple[float, ...]]:
    """提取 baseline 输入中各个“越大越好”质量效用的阈值样本。"""

    result: dict[str, list[float]] = {}
    for entry in entries:
        if entry.status != "observed":
            continue
        for name, value in quality_utility_values(entry, settings).items():
            if name == "brightness":
                continue
            result.setdefault(name, []).append(value)
    return {name: tuple(values) for name, values in result.items()}


def _error_vs_reject_report(
    records_by_name: dict[str, list[QualityUtilityRecord]],
    clean_values_by_name: dict[str, tuple[float, ...]],
) -> dict[str, object]:
    """为每项质量效用生成拒绝曲线点和等频错误率分桶。"""

    report: dict[str, object] = {}
    for name, records in sorted(records_by_name.items()):
        report[name] = {
            "points": [
                asdict(point)
                for point in error_vs_reject(
                    records,
                    clean_values=clean_values_by_name[name],
                    retention_targets=(0.99, 0.95, 0.90),
                )
            ],
            "error_bins_low_to_high": [asdict(item) for item in quality_error_bins(records)],
        }
    return report


def _quality_configuration(settings: Settings) -> dict[str, float | int]:
    """返回本报告实际复算的完整质量配置。"""

    return {
        "min_detection_score": settings.min_detection_score,
        "min_face_size_px": settings.min_face_size_px,
        "min_blur_variance": settings.min_blur_variance,
        "min_brightness": settings.min_brightness,
        "max_brightness": settings.max_brightness,
        "min_contrast": settings.min_contrast,
        "high_quality_score": settings.high_quality_score,
        "medium_quality_score": settings.medium_quality_score,
    }


if __name__ == "__main__":
    raise SystemExit(main())
