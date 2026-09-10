from __future__ import annotations

import cv2
import numpy as np

from camera_face_comparison.experiment_artifacts import file_sha256
from camera_face_comparison.raw_embedding_cache import RawEmbeddingCache
from camera_face_comparison.xqlfw import load_xqlfw_protocol, load_xqlfw_quality_scores
from camera_face_comparison.xqlfw_evaluation import evaluate_xqlfw_protocol


def test_xqlfw_parser_expands_all_folds_and_resolves_images(tmp_path) -> None:
    """XQLFW 标准 pairs 文件必须展开为正负样本并检查图片路径。"""

    dataset_dir = tmp_path / "lfw"
    for identity, indices in {"Alice": (1, 2), "Bob": (1,), "Carol": (1,)}.items():
        identity_dir = dataset_dir / identity
        identity_dir.mkdir(parents=True)
        for index in indices:
            frame = np.full((240, 320, 3), 100 if identity == "Alice" else 200, dtype=np.uint8)
            assert cv2.imwrite(str(identity_dir / f"{identity}_{index:04d}.jpg"), frame)

    pairs_path = tmp_path / "pairs.txt"
    pairs_path.write_text(
        "1 1\nAlice 1 2\nAlice 1 Bob 1\n",
        encoding="utf-8",
    )
    protocol = load_xqlfw_protocol(pairs_path, dataset_dir)

    assert protocol.fold_count == 1
    assert protocol.pairs_per_fold == 1
    assert len(protocol.pairs) == 2
    assert protocol.pairs[0].same_identity is True
    assert protocol.pairs[1].same_identity is False
    assert protocol.image_paths == (
        "Alice/Alice_0001.jpg",
        "Alice/Alice_0002.jpg",
        "Bob/Bob_0001.jpg",
    )


def test_xqlfw_quality_score_parser_uses_image_relative_paths(tmp_path) -> None:
    """官方质量分必须转换为与 pairs 协议一致的图片相对路径。"""

    scores_path = tmp_path / "scores.txt"
    scores_path.write_text(
        "ID\tNum\tScore\nAlice\t1\t0.25\nAlice\t2\t0.75\n",
        encoding="utf-8",
    )

    scores = load_xqlfw_quality_scores(scores_path)

    assert scores == {
        "Alice/Alice_0001.jpg": 0.25,
        "Alice/Alice_0002.jpg": 0.75,
    }


def test_xqlfw_evaluation_uses_fold_calibration_and_raw_cache(tmp_path) -> None:
    """每一折必须只用其他折选阈值，并在原始缓存的有效图片上评测。"""

    dataset_dir = tmp_path / "lfw"
    for identity in ("Alice", "Bob"):
        identity_dir = dataset_dir / identity
        identity_dir.mkdir(parents=True)
        for index in (1, 2):
            frame = np.full((32, 32, 3), 100 + index, dtype=np.uint8)
            assert cv2.imwrite(str(identity_dir / f"{identity}_{index:04d}.jpg"), frame)
    pairs_path = tmp_path / "pairs.txt"
    pairs_path.write_text(
        "2 1\n"
        "Alice 1 2\nAlice 1 Bob 1\n"
        "Bob 1 2\nAlice 2 Bob 2\n",
        encoding="utf-8",
    )
    protocol = load_xqlfw_protocol(pairs_path, dataset_dir)
    embeddings = {
        "Alice/Alice_0001.jpg": np.array([1.0, 0.0], dtype=np.float32),
        "Alice/Alice_0002.jpg": np.array([0.9, 0.1], dtype=np.float32),
        "Bob/Bob_0001.jpg": np.array([0.0, 1.0], dtype=np.float32),
        "Bob/Bob_0002.jpg": np.array([0.1, 0.9], dtype=np.float32),
    }
    quality_scores = {path: 0.2 + index * 0.1 for index, path in enumerate(embeddings)}

    with RawEmbeddingCache(tmp_path / "raw.sqlite", "xqlfw", "model-a") as cache:
        for relative_path, embedding in embeddings.items():
            cache.put_observed(
                relative_path=relative_path,
                file_sha256=file_sha256(dataset_dir / relative_path),
                embedding=embedding,
                metrics={"face_size_px": 100.0},
                face_count=1,
                latency_ms=2.0,
            )
        cache.commit()
        result = evaluate_xqlfw_protocol(
            dataset_dir=dataset_dir,
            protocol=protocol,
            quality_scores=quality_scores,
            cache=cache,
        )

    assert result.image_total == 4
    assert result.valid_image_total == 4
    assert result.pair_total == 4
    assert result.valid_pair_total == 4
    assert result.positive_correct == 2
    assert result.negative_correct == 2
    assert result.accuracy == 1.0
    assert result.rejected_images == ()
    assert len(result.folds) == 2
    np.testing.assert_allclose(
        sorted(pair.min_quality for pair in result.pairs),
        [0.2, 0.2, 0.3, 0.4],
    )
