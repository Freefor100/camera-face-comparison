from __future__ import annotations

import cv2
import numpy as np

from camera_face_comparison.face_engine import FaceObservation
from camera_face_comparison.raw_embedding_cache import RawEmbeddingCache
from camera_face_comparison.raw_lfw_extraction import extract_or_load_raw_embedding


class MultiFaceEngine:
    """返回一张主体脸和一张背景小脸的有标签数据集测试引擎。"""

    def __init__(self) -> None:
        """初始化模型调用计数。"""
        self.calls = 0

    def extract_faces(self, frame: np.ndarray) -> list[FaceObservation]:
        """返回面积不同的两张脸，主体脸 embedding 为第一个坐标轴。"""

        self.calls += 1
        return [
            FaceObservation(
                bbox=(10.0, 10.0, 190.0, 190.0),
                detection_score=0.90,
                embedding=np.array([3.0, 0.0], dtype=np.float32),
                blur_variance=120.0,
                landmarks=None,
            ),
            FaceObservation(
                bbox=(200.0, 20.0, 240.0, 60.0),
                detection_score=0.99,
                embedding=np.array([0.0, 4.0], dtype=np.float32),
                blur_variance=80.0,
                landmarks=None,
            ),
        ]


def test_raw_cache_restores_embedding_and_metrics_without_quality_policy(tmp_path) -> None:
    """原始缓存只能由图片、数据集和提取版本失效，不能包含质量策略。"""

    path = tmp_path / "raw.sqlite"
    first = RawEmbeddingCache(path, "lfw", "model-a")
    first.put_observed(
        relative_path="Alice/Alice_0001.jpg",
        file_sha256="hash-a",
        embedding=np.array([1.0, 0.0], dtype=np.float32),
        metrics={"detection_score": 0.9},
        face_count=1,
        latency_ms=12.0,
    )
    first.commit()
    first.close()

    second = RawEmbeddingCache(path, "lfw", "model-a")
    entry = second.get("Alice/Alice_0001.jpg", "hash-a")

    assert entry is not None
    assert entry.status == "observed"
    np.testing.assert_array_equal(entry.embedding, np.array([1.0, 0.0], dtype=np.float32))
    assert entry.metrics == {"detection_score": 0.9}
    assert second.get("Alice/Alice_0001.jpg", "changed-hash") is None
    second.close()


def test_labeled_dataset_extraction_selects_primary_face_and_reuses_raw_cache(tmp_path) -> None:
    """有标签数据集中的背景小脸不能触发桌面应用的多人脸拒绝。"""

    image_path = tmp_path / "Alice_0001.jpg"
    frame = np.full((250, 250, 3), 120, dtype=np.uint8)
    frame[:, ::2] = 160
    assert cv2.imwrite(str(image_path), frame)
    engine = MultiFaceEngine()
    with RawEmbeddingCache(tmp_path / "raw.sqlite", "lfw", "model-a") as cache:
        first = extract_or_load_raw_embedding(
            image_path=image_path,
            relative_path="Alice/Alice_0001.jpg",
            face_engine=engine,
            cache=cache,
        )
        second = extract_or_load_raw_embedding(
            image_path=image_path,
            relative_path="Alice/Alice_0001.jpg",
            face_engine=engine,
            cache=cache,
        )

    assert engine.calls == 1
    assert first.face_count == 2
    np.testing.assert_array_equal(first.embedding, np.array([1.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(second.embedding, first.embedding)
