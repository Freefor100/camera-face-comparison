from __future__ import annotations

import argparse
import json
from pathlib import Path

from camera_face_comparison.config import load_settings
from camera_face_comparison.experiment_artifacts import write_json_atomic
from camera_face_comparison.face_engine import FaceEngine
from camera_face_comparison.face_library import InMemoryFaceLibrary
from camera_face_comparison.image_input import ImageInput
from camera_face_comparison.integrity import verify_library
from camera_face_comparison.recognition import RecognitionService
from camera_face_comparison.repository import FaceRepository


def parse_args() -> argparse.Namespace:
    """读取最终演示标准库自动验收参数。"""
    parser = argparse.ArgumentParser(description="验收最终演示标准库的 Known、Unknown 和重启恢复")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--known",
        type=Path,
        default=Path("data/demo-candidates/barack_obama/barack_obama_2012_original.jpg"),
    )
    parser.add_argument(
        "--unknown",
        type=Path,
        default=Path("data/phase1-inputs/joe_biden_official.jpg"),
    )
    parser.add_argument(
        "--expected-known-name", default="Barack Obama"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/experiments/phase6/final-acceptance-report.json"),
    )
    return parser.parse_args()


def main() -> int:
    """运行最终演示库的完整性、Known、Unknown 和重启恢复验收。"""
    args = parse_args()
    data_dir = args.data_dir.resolve()
    settings = load_settings(data_dir)
    repository = FaceRepository(settings.database_path)
    try:
        people = repository.list_people()
        samples = repository.list_samples()
        initial_integrity = verify_library(repository, settings)
        if not initial_integrity.is_valid:
            raise RuntimeError(f"initial demo gallery integrity failed: {initial_integrity.failures[0]}")

        library = InMemoryFaceLibrary.from_repository(repository)
        engine = FaceEngine.from_local_model(settings)
        known_result = _recognize(
            repository, settings, engine, library.snapshot(), args.known
        )
        unknown_result = _recognize(
            repository, settings, engine, library.snapshot(), args.unknown
        )
        restart_snapshot = None
    finally:
        repository.close()

    restarted_repository = FaceRepository(settings.database_path)
    try:
        restarted_integrity = verify_library(restarted_repository, settings)
        restarted_library = InMemoryFaceLibrary.from_repository(restarted_repository)
        restart_snapshot = restarted_library.snapshot()
        restarted_people_count = len(restarted_repository.list_people())
        restarted_sample_count = len(restarted_repository.list_samples())
    finally:
        restarted_repository.close()

    report = {
        "artifact": "phase6-final-acceptance-v1",
        "model_backend": {
            "name": engine.backend.name,
            "providers": list(engine.backend.providers),
        },
        "gallery": {
            "people_count": len(people),
            "sample_count": len(samples),
            "person_names": [person.display_name for person in people],
        },
        "initial_integrity": {"valid": initial_integrity.is_valid},
        "known": _result_payload(known_result, args.expected_known_name),
        "unknown": _result_payload(unknown_result, None),
        "restart": {
            "integrity_valid": restarted_integrity.is_valid,
            "people_count": restarted_people_count,
            "sample_count": restarted_sample_count,
            "snapshot_people_count": len(restart_snapshot.person_ids),
        },
        "acceptance": {
            "known_name_correct": (
                known_result.status == "matched"
                and known_result.display_name == args.expected_known_name
            ),
            "unknown_rejected": unknown_result.status == "unknown",
            "restart_recovered": (
                restarted_integrity.is_valid
                and restarted_people_count == len(people)
                and restarted_sample_count == len(samples)
                and len(restart_snapshot.person_ids) == len(library.snapshot().person_ids)
            ),
        },
    }
    report["acceptance"]["all_passed"] = all(report["acceptance"].values())
    write_json_atomic(args.output.resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["acceptance"]["all_passed"] else 1


def _recognize(repository, settings, engine, snapshot, path: Path):
    """使用正式识别服务处理一张本地图片。"""
    image_input = ImageInput.from_file(path, source_type="file")
    return RecognitionService(repository, settings, engine, snapshot).compare_input(image_input)


def _result_payload(result, expected_name: str | None) -> dict[str, object]:
    """提取验收报告所需的识别结果字段，不保存图片或 embedding。"""
    return {
        "status": result.status,
        "display_name": result.display_name,
        "top_score": result.top_score,
        "second_score": result.second_score,
        "score_gap": result.score_gap,
        "reason": result.reason,
        "latency_ms": result.latency_ms,
        "expected_name": expected_name,
    }


if __name__ == "__main__":
    raise SystemExit(main())
