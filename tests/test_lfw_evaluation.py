from __future__ import annotations

import cv2
import numpy as np

from camera_face_comparison.config import load_settings
from camera_face_comparison.face_engine import FaceObservation
from camera_face_comparison.lfw_dataset import LfwProbe, LfwProtocol
from camera_face_comparison.lfw_evaluation import evaluate_lfw_protocol


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
