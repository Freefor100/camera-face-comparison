from __future__ import annotations

import cv2
import numpy as np

from camera_face_comparison.config import load_settings
from camera_face_comparison.evaluation_cache import EvaluationEmbeddingCache
from camera_face_comparison.face_engine import FaceObservation
from camera_face_comparison.lfw_dataset import LfwProbe, LfwProtocol
from camera_face_comparison.lfw_evaluation import (
    evaluate_lfw_protocol,
    evaluate_lfw_protocol_streaming,
)


class MarkerFaceEngine:
    """根据测试图片像素标记返回预设特征的评测引擎。"""

    def extract_single_face(self, frame: np.ndarray) -> FaceObservation:
        """把图片左上角的标记映射到预设的人脸特征。"""
        marker = int(frame[0, 0, 0])
        if marker <= 140:
            embedding = np.array([1.0, 0.0], dtype=np.float32)
        elif marker >= 220:
            embedding = np.array([0.0, 1.0], dtype=np.float32)
        else:
            embedding = np.array([0.70710677, 0.70710677], dtype=np.float32)
        return FaceObservation(
            bbox=(0.0, 0.0, 180.0, 180.0),
            detection_score=0.95,
            embedding=embedding,
            blur_variance=150.0,
            landmarks=None,
        )


class CountingMarkerFaceEngine(MarkerFaceEngine):
    """记录调用次数、用于验证评测缓存是否生效的测试引擎。"""

    def __init__(self) -> None:
        """初始化调用计数器。"""
        self.calls = 0

    def extract_single_face(self, frame: np.ndarray) -> FaceObservation:
        """记录一次模型调用后复用标记引擎的特征结果。"""
        self.calls += 1
        return super().extract_single_face(frame)


def test_lfw_evaluation_extracts_templates_and_keeps_unknown_probe_labels(tmp_path) -> None:
    """真实图片评测必须独立构建标准库，并保留未知探针标签。"""

    dataset_dir = tmp_path / "lfw_funneled"
    enrollment = {
        "Alice": tuple(_write_images(dataset_dir, "Alice", 100, 1)),
        "Bob": tuple(_write_images(dataset_dir, "Bob", 200, 1)),
    }
    known_probe = _write_images(dataset_dir, "Alice", 100, 1, start=2)[0]
    unknown_probe = _write_images(dataset_dir, "Unknown", 145, 1)[0]
    protocol = LfwProtocol(
        enrollment=enrollment,
        probes=(LfwProbe(known_probe, "Alice"), LfwProbe(unknown_probe, None)),
    )

    result = evaluate_lfw_protocol(
        dataset_dir=dataset_dir,
        protocol=protocol,
        settings=load_settings(tmp_path / "data"),
        face_engine=MarkerFaceEngine(),
    )

    assert result.gallery_person_ids == ("Alice", "Bob")
    assert result.enrollment_rejections == ()
    assert result.probe_rejections == ()
    assert [record.expected_person_id for record in result.records] == ["Alice", None]
    assert set(result.records[0].sample_scores) == {"Alice", "Bob"}
    assert [record.expected_person_id for record in result.embedding_records] == ["Alice", None]
    assert set(result.embedding_records[0].gallery_embeddings) == {"Alice", "Bob"}


def test_streaming_lfw_evaluation_processes_all_probes_without_retaining_scores(tmp_path) -> None:
    """全量评测接口必须统计所有探针，并只保留汇总结果和拒绝原因。"""

    dataset_dir = tmp_path / "lfw_funneled"
    alice_gallery = _write_images(dataset_dir, "Alice", 100, 1)
    bob_gallery = _write_images(dataset_dir, "Bob", 200, 1)
    alice_probe = _write_images(dataset_dir, "Alice", 100, 1, start=2)[0]
    unknown_probe = _write_images(dataset_dir, "Unknown", 145, 1)[0]
    protocol = LfwProtocol(
        enrollment={"Alice": tuple(alice_gallery), "Bob": tuple(bob_gallery)},
        probes=(LfwProbe(alice_probe, "Alice"), LfwProbe(unknown_probe, None)),
    )
    progress: list[tuple[str, int, int]] = []

    result = evaluate_lfw_protocol_streaming(
        dataset_dir=dataset_dir,
        protocol=protocol,
        settings=load_settings(tmp_path / "data"),
        face_engine=MarkerFaceEngine(),
        methods=("max", "mean_prototype"),
        on_progress=lambda stage, done, total: progress.append((stage, done, total)),
    )

    assert result.gallery_image_total == 2
    assert result.gallery_valid_image_total == 2
    assert result.probe_total == 2
    assert result.probe_valid_image_total == 2
    assert result.enrollment_rejections == ()
    assert result.probe_rejections == ()
    assert result.methods["max"].known_correct == 1
    assert result.methods["max"].unknown_rejected == 1
    assert result.methods["mean_prototype"].known_correct == 1
    assert progress[-1] == ("probe", 2, 2)


def test_streaming_lfw_evaluation_reuses_cached_embeddings(tmp_path) -> None:
    """重新运行同一协议时，已缓存图片不能再次调用特征引擎。"""

    dataset_dir = tmp_path / "lfw_funneled"
    gallery = _write_images(dataset_dir, "Alice", 100, 1)
    probe = _write_images(dataset_dir, "Alice", 100, 1, start=2)
    protocol = LfwProtocol(
        enrollment={"Alice": tuple(gallery)},
        probes=(LfwProbe(probe[0], "Alice"),),
    )
    engine = CountingMarkerFaceEngine()
    cache_path = tmp_path / "cache.sqlite"

    with EvaluationEmbeddingCache(cache_path, "lfw", "buffalo_l:80") as cache:
        evaluate_lfw_protocol_streaming(
            dataset_dir=dataset_dir,
            protocol=protocol,
            settings=load_settings(tmp_path / "data"),
            face_engine=engine,
            methods=("max",),
            cache=cache,
        )
        first_run_calls = engine.calls
        evaluate_lfw_protocol_streaming(
            dataset_dir=dataset_dir,
            protocol=protocol,
            settings=load_settings(tmp_path / "data"),
            face_engine=engine,
            methods=("max",),
            cache=cache,
        )

    assert first_run_calls == 2
    assert engine.calls == first_run_calls


def _write_images(
    dataset_dir,
    name: str,
    base_value: int,
    count: int,
    *,
    start: int = 1,
) -> list[str]:
    """生成带身份和顺序文件名的测试 PNG 图片并返回相对路径。"""
    person_dir = dataset_dir / name
    person_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for index in range(start, start + count):
        frame = np.full((240, 320, 3), base_value, dtype=np.uint8)
        frame[:, ::2] = base_value + 30
        path = person_dir / f"{name}_{index:04d}.png"
        assert cv2.imwrite(str(path), frame)
        paths.append(path.relative_to(dataset_dir).as_posix())
    return paths
