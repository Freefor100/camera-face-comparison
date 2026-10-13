"""汇总既有短回放原始记录，不重新推理或覆盖历史实验。"""

import argparse
import hashlib
import json
import platform
import subprocess
from collections import Counter
from pathlib import Path

from experiments.commands.benchmark_refactor import summarize
from experiments.face_evaluation.experiment_artifacts import write_json_atomic


def compare_results(first, second):
    """核对同一工作负载的身份、判定和选帧，并单独报告数值误差。"""
    if first["workload"] != second["workload"]:
        raise ValueError("工作负载不同，不能做等价比较。")
    count, mismatches, errors = 0, [], []
    for scale, cases in first["cases"].items():
        for mode in ("single", "camera"):
            rows, other = cases[mode], second["cases"][scale][mode]
            if len(rows) != len(other):
                raise ValueError("记录数量不同，不能做等价比较。")
            for index, (before, after) in enumerate(zip(rows, other, strict=True)):
                count += 1
                fields = ("status", "person_id", "valid_frame_count", "selected_frame_index")
                if any(before[key] != after[key] for key in fields):
                    mismatches.append({"scale": scale, "mode": mode, "index": index})
                for key in ("top_score", "second_score"):
                    if before[key] is None or after[key] is None:
                        if before[key] != after[key]:
                            mismatches.append(
                                {"scale": scale, "mode": mode, "index": index, "field": key}
                            )
                    else:
                        errors.append(abs(before[key] - after[key]))
    return {
        "record_count": count,
        "mismatches": mismatches,
        "maximum_score_error": max(errors, default=0.0),
    }


def summarize_report(report):
    """各阶段独立统计；不把重叠阶段的耗时相加当作总用时。"""
    cases = {}
    for scale, values in report["cases"].items():
        cases[scale] = {}
        for mode in ("single", "camera"):
            rows = values[mode]
            cases[scale][mode] = {
                "record_count": len(rows),
                "status_counts": dict(Counter(row["status"] for row in rows)),
                "stages": {
                    key: summarize([row[key] for row in rows])
                    for key in rows[0]
                    if key.endswith("_ms")
                },
            }
    return {"model_load_ms": report["model_load_ms"], "backend": report["backend"], "cases": cases}


def main():
    """读取本轮四份报告及加载、缩略图记录，生成可追溯比较。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/experiments/refactor"))
    args = parser.parse_args()
    names = ("baseline", "optimized", "cpu-validation", "cpu-reference")
    reports = {name: json.loads((args.root / f"{name}.json").read_text()) for name in names}
    thumbnail = json.loads((args.root / "thumbnail-decoding.json").read_text())
    loading = json.loads((args.root / "library-loading-repeated.json").read_text())
    comparison = {
        "summary_code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "source_report_sha256": {
            name: hashlib.sha256((args.root / f"{name}.json").read_bytes()).hexdigest()
            for name in names
        },
        "protocol_sha256": reports["baseline"]["protocol_sha256"],
        "summaries": {name: summarize_report(report) for name, report in reports.items()},
        "cuda_before_after": compare_results(reports["baseline"], reports["optimized"]),
        "cpu_before_after": compare_results(reports["cpu-reference"], reports["cpu-validation"]),
        "cpu_cuda_numeric_difference": compare_results(
            reports["cpu-validation"], reports["optimized"]
        ),
        "library_loading": loading,
        "thumbnail_decoding": {
            **thumbnail,
            "full_summary": summarize(thumbnail["full_decode_ms"]),
            "scaled_summary": summarize(thumbnail["scaled_decode_ms"]),
        },
        "measurement_boundary": "输入解码单列；服务总用时含帧窗口、推理、检索、判定与日志，不含实际摄像头驱动、Qt 往返和页面生成。模型加载为单次进程观测。",
    }
    write_json_atomic(args.root / "comparison.json", comparison)
    print(
        json.dumps(
            {
                key: comparison[key]
                for key in ("cuda_before_after", "cpu_before_after", "cpu_cuda_numeric_difference")
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
