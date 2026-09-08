from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from camera_face_comparison.config import load_settings
from camera_face_comparison.decision_scores import (
    export_lfw_decision_scores,
    summarize_decision_scores,
)
from camera_face_comparison.evaluation_cache import (
    EvaluationEmbeddingCache,
    available_cache_extraction_ids,
    embedding_extraction_id,
    file_sha256,
    write_json_atomic,
)
from camera_face_comparison.lfw_dataset import (
    read_lfw_protocol,
    split_lfw_protocol,
    write_lfw_split_protocol,
)


def main() -> int:
    """从现有 LFW 缓存固定分区并导出不含判定阈值的六种方法分数。"""

    parser = argparse.ArgumentParser(
        description="Export threshold-free LFW identity scores from an existing embedding cache."
    )
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--source-protocol", type=Path)
    parser.add_argument("--cache-path", type=Path)
    parser.add_argument("--cache-extraction-id")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--calibration-fraction", type=float, default=0.5)
    parser.add_argument("--min-face-size", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    settings = load_settings(args.data_dir)
    source_protocol_path = (
        args.source_protocol
        or settings.data_dir / "datasets" / "lfw_full_open_set_protocol.json"
    )
    cache_path = args.cache_path or settings.logs_dir / "cache" / "lfw.sqlite"
    output_dir = args.output_dir or settings.data_dir / "experiments" / "phase2"
    dataset_dir = settings.data_dir / "datasets" / "lfw_funneled"
    try:
        source_protocol = read_lfw_protocol(source_protocol_path)
        split_protocol = split_lfw_protocol(
            source_protocol,
            seed=args.seed,
            calibration_fraction=args.calibration_fraction,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        split_protocol_path = output_dir / "protocol.json"
        write_lfw_split_protocol(split_protocol, split_protocol_path)
        split_protocol_sha256 = file_sha256(split_protocol_path)
        cache_extraction_id = _select_cache_extraction_id(
            cache_path,
            dataset_id="lfw-full-open-set-v1",
            requested=args.cache_extraction_id,
        )
        run_id = f"lfw-phase2-{split_protocol_sha256[:12]}-{_safe_id(cache_extraction_id)}"
        with EvaluationEmbeddingCache(
            cache_path,
            "lfw-full-open-set-v1",
            cache_extraction_id,
        ) as cache:
            summary = export_lfw_decision_scores(
                dataset_dir=dataset_dir,
                protocol=split_protocol,
                cache=cache,
                output_path=output_dir / "decision_scores.sqlite",
                run_id=run_id,
                protocol_sha256=split_protocol_sha256,
                embedding_extraction_id=cache_extraction_id,
                batch_size=args.batch_size,
                on_progress=_print_progress,
            )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1

    extraction_settings = replace(settings, min_face_size_px=args.min_face_size)
    manifest = {
        "artifact": "lfw-threshold-free-decision-scores-v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "run_id": run_id,
        "source_protocol": str(source_protocol_path),
        "source_protocol_sha256": split_protocol.source_protocol_sha256,
        "split_protocol": str(split_protocol_path),
        "split_protocol_sha256": split_protocol_sha256,
        "random_seed": args.seed,
        "calibration_fraction": args.calibration_fraction,
        "cache_path": str(cache_path),
        "cache_dataset_id": "lfw-full-open-set-v1",
        "selected_cache_extraction_id": cache_extraction_id,
        "current_embedding_extraction_id": embedding_extraction_id(extraction_settings),
        "quality_configuration": {
            "min_detection_score": extraction_settings.min_detection_score,
            "min_face_size_px": extraction_settings.min_face_size_px,
            "min_blur_variance": extraction_settings.min_blur_variance,
            "min_brightness": extraction_settings.min_brightness,
            "max_brightness": extraction_settings.max_brightness,
            "min_contrast": extraction_settings.min_contrast,
            "high_quality_score": extraction_settings.high_quality_score,
            "medium_quality_score": extraction_settings.medium_quality_score,
        },
        "aggregation_variants": [
            {"method": method, "top_k": top_k}
            for method, top_k in (
                ("single", 0),
                ("max", 0),
                ("mean_prototype", 0),
                ("top_k_mean", 2),
                ("top_k_mean", 3),
                ("top_k_mean", 5),
            )
        ],
        "coverage": asdict(summary),
        "code": _code_version(),
        "notes": [
            "decision_scores.sqlite 不保存 match_threshold 或 min_score_gap。",
            "selected_cache_extraction_id 是本次实际读取的既有缓存批次标识。",
            "current_embedding_extraction_id 用于采用拆分标识后的后续特征提取。",
        ],
    }
    write_json_atomic(
        output_dir / "calibration_summary.json",
        summarize_decision_scores(
            output_dir / "decision_scores.sqlite",
            run_id=run_id,
            split="calibration",
        ),
    )
    write_json_atomic(
        output_dir / "evaluation_summary.json",
        summarize_decision_scores(
            output_dir / "decision_scores.sqlite",
            run_id=run_id,
            split="evaluation",
        ),
    )
    write_json_atomic(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


def _select_cache_extraction_id(path: Path, *, dataset_id: str, requested: str | None) -> str:
    """选择明确的缓存批次；多批次并存时要求调用者指定。"""

    available = available_cache_extraction_ids(path, dataset_id)
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


def _safe_id(value: str) -> str:
    """把缓存标识转换为适合放入运行编号的短字符串。"""

    return "".join(character if character.isalnum() else "-" for character in value)[-24:]


def _code_version() -> dict[str, object]:
    """记录当前 Git 提交和工作区状态，失败时保留明确的 unknown。"""

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {
            "commit": "unknown",
            "dirty": None,
            "source_tree_sha256": _source_tree_sha256(),
        }
    return {
        "commit": commit,
        "dirty": bool(status.strip()),
        "source_tree_sha256": _source_tree_sha256(),
    }


def _source_tree_sha256() -> str:
    """计算当前 Python 源码和项目配置的内容哈希，覆盖未提交文件。"""

    digest = hashlib.sha256()
    paths = sorted((PROJECT_ROOT / "src").rglob("*.py"))
    paths.extend(sorted((PROJECT_ROOT / "scripts").glob("*.py")))
    paths.append(PROJECT_ROOT / "pyproject.toml")
    for path in paths:
        relative_path = path.relative_to(PROJECT_ROOT).as_posix()
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _print_progress(stage: str, done: int, total: int) -> None:
    """以低频率显示缓存读取和矩阵打分进度。"""

    if done == 1 or done == total or done % 500 == 0:
        print(f"Phase 2 {stage}: {done}/{total}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
