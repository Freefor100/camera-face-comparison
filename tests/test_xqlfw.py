from __future__ import annotations

import cv2
import numpy as np

from camera_face_comparison.config import load_settings
from camera_face_comparison.face_engine import FaceObservation
from camera_face_comparison.xqlfw import load_xqlfw_protocol
from camera_face_comparison.xqlfw_evaluation import evaluate_xqlfw_protocol


class MarkerFaceEngine:
    """根据图片像素标记返回测试用特征。"""

    def extract_single_face(self, frame: np.ndarray) -> FaceObservation:
        """把图片左上角的像素标记转换为单位特征向量。"""
        marker = int(frame[0, 0, 0])
        embedding = (
            np.array([1.0, 0.0], dtype=np.float32)
            if marker < 150
            else np.array([0.0, 1.0], dtype=np.float32)
        )
        return FaceObservation(
            bbox=(0.0, 0.0, 180.0, 180.0),
            detection_score=0.95,
            embedding=embedding,
            blur_variance=150.0,
            landmarks=None,
        )


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


def test_xqlfw_evaluation_scores_every_valid_pair(tmp_path) -> None:
    """XQLFW 验证评测必须在同一批向量上统计正负对准确率。"""

    dataset_dir = tmp_path / "lfw"
    for identity, marker in (("Alice", 100), ("Bob", 200)):
        identity_dir = dataset_dir / identity
        identity_dir.mkdir(parents=True)
        for index in (1, 2):
            frame = np.full((240, 320, 3), marker, dtype=np.uint8)
            frame[:, ::2] = min(marker + 30, 255)
            assert cv2.imwrite(str(identity_dir / f"{identity}_{index:04d}.jpg"), frame)
    pairs_path = tmp_path / "pairs.txt"
    pairs_path.write_text(
        "1 1\nAlice 1 2\nAlice 1 Bob 1\n",
        encoding="utf-8",
    )
    protocol = load_xqlfw_protocol(pairs_path, dataset_dir)

    result = evaluate_xqlfw_protocol(
        dataset_dir=dataset_dir,
        protocol=protocol,
        settings=load_settings(tmp_path / "data"),
        face_engine=MarkerFaceEngine(),
        threshold=0.5,
    )

    assert result.image_total == 3
    assert result.valid_image_total == 3
    assert result.pair_total == 2
    assert result.valid_pair_total == 2
    assert result.positive_correct == 1
    assert result.negative_correct == 1
    assert result.accuracy == 1.0
    assert result.rejected_images == ()
