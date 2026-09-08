from __future__ import annotations

import numpy as np

from camera_face_comparison.image_input import QualityProfile


def test_evaluation_cache_starts_empty(tmp_path) -> None:
    """新的评测缓存不能伪造已经处理过的图片。"""

    from camera_face_comparison.evaluation_cache import EvaluationEmbeddingCache

    cache = EvaluationEmbeddingCache(tmp_path / "cache.sqlite", "lfw", "buffalo_l:80")

    assert cache.get("Alice/Alice_0001.jpg", "hash-1") is None
    cache.close()


def test_evaluation_cache_restores_valid_embedding_after_restart(tmp_path) -> None:
    """有效 embedding 和质量信息提交后，重新打开缓存仍可读取。"""

    from camera_face_comparison.evaluation_cache import EvaluationEmbeddingCache

    cache_path = tmp_path / "cache.sqlite"
    profile = QualityProfile("high", 0.82, {"brightness": 120.0}, ())
    first = EvaluationEmbeddingCache(cache_path, "lfw", "buffalo_l:80")
    first.put_valid("Alice/Alice_0001.jpg", "hash-1", np.array([1.0, 2.0]), profile)
    first.commit()
    first.close()

    second = EvaluationEmbeddingCache(cache_path, "lfw", "buffalo_l:80")
    entry = second.get("Alice/Alice_0001.jpg", "hash-1")

    assert entry is not None
    assert entry.embedding is not None
    np.testing.assert_array_equal(entry.embedding, np.array([1.0, 2.0], dtype=np.float32))
    assert entry.quality == profile
    assert entry.reason is None
    second.close()


def test_evaluation_cache_restores_rejection_reason(tmp_path) -> None:
    """拒绝项也必须缓存，避免重启后重复处理同一失败图片。"""

    from camera_face_comparison.evaluation_cache import EvaluationEmbeddingCache

    cache = EvaluationEmbeddingCache(tmp_path / "cache.sqlite", "lfw", "buffalo_l:80")
    cache.put_rejected("Bob/Bob_0001.jpg", "hash-2", "multiple_faces")
    cache.commit()

    entry = cache.get("Bob/Bob_0001.jpg", "hash-2")

    assert entry is not None
    assert entry.embedding is None
    assert entry.reason == "multiple_faces"
    cache.close()


def test_evaluation_cache_invalidates_changed_file_and_model(tmp_path) -> None:
    """图片摘要或模型配置变化时不能复用旧 embedding。"""

    from camera_face_comparison.evaluation_cache import EvaluationEmbeddingCache

    cache_path = tmp_path / "cache.sqlite"
    first = EvaluationEmbeddingCache(cache_path, "lfw", "buffalo_l:80")
    first.put_valid(
        "Alice/Alice_0001.jpg",
        "hash-1",
        np.array([1.0, 0.0]),
        QualityProfile("high", 0.8, {}, ()),
    )
    first.commit()
    assert first.get("Alice/Alice_0001.jpg", "hash-2") is None
    first.close()

    changed_model = EvaluationEmbeddingCache(cache_path, "lfw", "buffalo_l:112")
    assert changed_model.get("Alice/Alice_0001.jpg", "hash-1") is None
    changed_model.close()
