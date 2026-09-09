from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from camera_face_comparison.config import load_settings
from camera_face_comparison.evaluation_cache import file_sha256, write_json_atomic
from camera_face_comparison.qmul_evaluation import (
    export_qmul_pressure_scores,
    summarize_qmul_pressure_scores,
)
from camera_face_comparison.qmul_survface import build_qmul_protocol
from camera_face_comparison.raw_embedding_cache import (
    RawEmbeddingCache,
    available_raw_extraction_ids,
)


def main() -> int:
    """从完整原始缓存导出并汇总 QMUL-SurvFace 监控域压力结果。"""

    parser = argparse.ArgumentParser(
        description="Evaluate QMUL-SurvFace from a complete policy-independent raw cache."
    )
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "datasets" / "qmul-survface" / "QMUL-SurvFace",
    )
    parser.add_argument("--cache-path", type=Path)
    parser.add_argument("--cache-extraction-id")
    parser.add_argument("--scores-output", type=Path)
    parser.add_argument("--report-output", type=Path)
    parser.add_argument("--transfer-policy", type=Path)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least one")

    settings = load_settings(args.data_dir)
    cache_path = args.cache_path or settings.logs_dir / "cache" / "qmul_survface_raw.sqlite"
    output_dir = settings.data_dir / "experiments" / "phase5"
    scores_path = args.scores_output or output_dir / "qmul_decision_scores.sqlite"
    report_path = args.report_output or output_dir / "qmul_pressure_report.json"
    policy_path = args.transfer_policy or (
        settings.data_dir / "experiments" / "phase4" / "final_evaluation.json"
    )
    dataset_dir = args.dataset_root / "Face_Identification_Test_Set"
    try:
        protocol = build_qmul_protocol(args.dataset_root)
        protocol_sha256 = _protocol_sha256(protocol)
        extraction_id = _select_cache_extraction_id(
            cache_path,
            dataset_id="qmul-survface-identification-raw-v1",
            requested=args.cache_extraction_id,
        )
        run_id = f"qmul-pressure-{protocol_sha256[:12]}-{_safe_id(extraction_id)}"
        with RawEmbeddingCache(
            cache_path,
            "qmul-survface-identification-raw-v1",
            extraction_id,
        ) as cache:
            coverage = export_qmul_pressure_scores(
                dataset_dir=dataset_dir,
                protocol=protocol,
                cache=cache,
                output_path=scores_path,
                run_id=run_id,
                protocol_sha256=protocol_sha256,
                embedding_extraction_id=extraction_id,
                batch_size=args.batch_size,
                on_progress=_print_progress,
            )
        transferred_policy = _read_lfw_policy(policy_path)
        report = summarize_qmul_pressure_scores(
            scores_path,
            run_id=run_id,
            transferred_policy=transferred_policy,
        ) | {
            "dataset_root": str(args.dataset_root),
            "cache_path": str(cache_path),
            "scores_path": str(scores_path),
            "coverage": asdict(coverage),
            "policy_source_sha256": file_sha256(policy_path),
            "notes": [
                "QMUL 不应用桌面启发式质量门，拒绝项只表示模型 FTE 或身份无有效 Gallery。",
                "Mean Prototype 来自 Phase 4 LFW 方法选择；QMUL 不重新比较聚合方法。",
                "LFW 工作点只做跨域迁移诊断，QMUL 标签未用于搜索阈值或候选分差。",
            ],
        }
        write_json_atomic(report_path, report)
    except (FileNotFoundError, KeyError, RuntimeError, TypeError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False))
    return 0


def _protocol_sha256(protocol) -> str:
    """对身份、角色和相对路径做确定性摘要，固定本次官方协议。"""

    digest = hashlib.sha256()
    for person_id in sorted(protocol.enrollment):
        for relative_path in protocol.enrollment[person_id]:
            digest.update(f"gallery\0{person_id}\0{relative_path}\n".encode())
    for probe in protocol.probes:
        role = "mated" if probe.expected_person_id is not None else "unmated"
        identity = probe.expected_person_id or ""
        digest.update(f"{role}\0{identity}\0{probe.relative_path}\n".encode())
    return digest.hexdigest()


def _select_cache_extraction_id(
    path: Path,
    *,
    dataset_id: str,
    requested: str | None,
) -> str:
    """选择唯一原始提取版本；多版本并存时要求显式指定。"""

    available = available_raw_extraction_ids(path, dataset_id)
    if requested is not None:
        if requested not in available:
            raise ValueError(
                f"cache extraction id is unavailable: {requested}; available={list(available)}"
            )
        return requested
    if len(available) != 1:
        raise ValueError(
            "--cache-extraction-id is required unless the cache contains exactly one id; "
            f"available={list(available)}"
        )
    return available[0]


def _read_lfw_policy(path: Path) -> dict[str, object]:
    """读取 Phase 4 唯一已选 LFW 工作点，拒绝其他聚合方法。"""

    payload = json.loads(path.read_text(encoding="utf-8"))
    selected = payload["selected"]
    point = selected["operating_point"]
    method = str(selected["method"])
    if method != "mean_prototype":
        raise ValueError("QMUL transfer diagnostic requires the selected Mean Prototype policy")
    return {
        "source": str(path),
        "method": method,
        "match_threshold": float(point["match_threshold"]),
        "use_score_gap": bool(point["use_score_gap"]),
        "min_score_gap": float(point["min_score_gap"]),
    }


def _safe_id(value: str) -> str:
    """把缓存版本转换成运行编号可用的短字符串。"""

    return "".join(character if character.isalnum() else "-" for character in value)[-24:]


def _print_progress(stage: str, done: int, total: int) -> None:
    """低频显示缓存校验与矩阵检索进度。"""

    if done == 1 or done == total or done % 10000 == 0:
        print(f"QMUL {stage}: {done}/{total}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
