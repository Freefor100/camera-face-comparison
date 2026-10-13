"""在临时 LFW 标准库中测量一致读取与矩阵构建，不修改演示库。"""

import argparse
import hashlib
import json
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np

from camera_face_comparison.config import load_settings
from camera_face_comparison.face_engine import normalize_embedding
from camera_face_comparison.face_library import InMemoryFaceLibrary
from camera_face_comparison.integrity import verify_library
from camera_face_comparison.repository import FaceRepository
from experiments.commands.benchmark_production_runtime import (
    _build_gallery_matrix,
    _read_cached_embeddings,
    _select_cache_partition,
)
from experiments.commands.benchmark_refactor import summarize
from experiments.face_evaluation.experiment_artifacts import write_json_atomic


def main():
    """记录同份图片下的旧逐人查询与新同源矩阵路径，避免伪造库大小。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/experiments/refactor/library-loading-repeated.json"),
    )
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("重复次数必须至少为一。")
    protocol = json.loads(Path("data/experiments/phase4/protocol.json").read_text())
    cache = Path("data/logs/cache/lfw_raw.sqlite")
    cached = _read_cached_embeddings(cache, *_select_cache_partition(cache))
    ids, _ = _build_gallery_matrix(protocol["enrollment"], cached, 4588)
    reports = {}
    for count in (3, 100, 4588):
        with tempfile.TemporaryDirectory(prefix="face-library-replay-") as root:
            settings = load_settings(Path(root))
            repository = FaceRepository(settings.database_path)
            timestamp = datetime.now(UTC).isoformat()
            rows = []
            for pid in ids[:count]:
                folder = settings.faces_dir / pid
                folder.mkdir()
                for index, relative in enumerate(protocol["enrollment"][pid]):
                    if relative not in cached:
                        continue
                    path = folder / f"sample_{index:03d}.jpg"
                    shutil.copyfile(Path("data/datasets/lfw_funneled") / relative, path)
                    vector = cached[relative].astype(np.float32)
                    rows.append(
                        (
                            f"{pid}-{index}",
                            pid,
                            path.relative_to(settings.data_dir).as_posix(),
                            vector.tobytes(),
                            512,
                            "{}",
                            timestamp,
                            "dataset",
                            hashlib.sha256(path.read_bytes()).hexdigest(),
                            hashlib.sha256(vector.tobytes()).hexdigest(),
                        )
                    )
            with repository._write_transaction():
                repository._connection.executemany(
                    "INSERT INTO persons VALUES (?,?,?)",
                    [(pid, pid, timestamp) for pid in ids[:count]],
                )
                repository._connection.executemany(
                    "INSERT INTO face_samples VALUES (?,?,?,?,?,?,?,?,?,?)", rows
                )
            measurements = {"integrity": [], "same_records_matrix": [], "per_person_queries": []}
            errors = []
            for _ in range(args.repeats):
                started = perf_counter()
                report = verify_library(repository, settings)
                measurements["integrity"].append((perf_counter() - started) * 1000)
                assert report.is_valid
                started = perf_counter()
                snapshot = InMemoryFaceLibrary.from_samples(
                    report.recognizable_people, report.samples
                ).snapshot()
                measurements["same_records_matrix"].append((perf_counter() - started) * 1000)
                started = perf_counter()
                old_entries = []
                # 这是旧查询策略的实验参考，仍使用当前严格解码器，不是旧版本总启动时间。
                for person in sorted(repository.list_people(), key=lambda p: p.id):
                    samples = repository.list_samples(person.id)
                    old_entries.append(
                        normalize_embedding(
                            np.mean([normalize_embedding(s.embedding) for s in samples], axis=0)
                        )
                    )
                measurements["per_person_queries"].append((perf_counter() - started) * 1000)
                errors.append(
                    float(np.max(np.abs(np.stack(old_entries) - snapshot.prototype_matrix)))
                )
            reports[str(count)] = {
                "sample_count": len(rows),
                "raw_ms": measurements,
                "summary": {key: summarize(values) for key, values in measurements.items()},
                "matrix_max_error": max(errors),
            }
            repository.close()
        print(f"已完成 {count} 人临时库的 {args.repeats} 次加载对照。", flush=True)
    write_json_atomic(args.output, reports)
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
