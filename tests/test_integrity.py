from __future__ import annotations

from pathlib import Path

import numpy as np

from camera_face_comparison.config import load_settings
from camera_face_comparison.enrollment import EnrollmentService
from camera_face_comparison.face_engine import FaceObservation
from camera_face_comparison.image_input import ImageInput
from camera_face_comparison.integrity import verify_library
from camera_face_comparison.repository import FaceRepository


class FakeFaceEngine:
    """返回固定合格观察结果的完整性测试引擎。"""

    def extract_single_face(self, frame: np.ndarray) -> FaceObservation:
        """为任意输入返回固定的人脸特征和质量指标。"""
        return FaceObservation(
            bbox=(0.0, 0.0, 180.0, 180.0),
            detection_score=0.95,
            embedding=np.array([0.2, 0.5, 0.8], dtype=np.float32),
            blur_variance=150.0,
            landmarks=None,
        )


def _save_image(path: Path, frame: np.ndarray) -> None:
    """保存完整性测试使用的样本文件占位内容。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"valid sample")


def test_library_verification_reports_a_replaced_sample_image(tmp_path) -> None:
    """应用外部修改样本文件后，完整性检查必须阻止标准库继续被信任。"""

    settings = load_settings(tmp_path)
    repository = FaceRepository(settings.database_path)
    service = EnrollmentService(
        repository=repository,
        settings=settings,
        face_engine=FakeFaceEngine(),
        image_saver=_save_image,
    )
    frame = np.full((240, 320, 3), 120, dtype=np.uint8)
    frame[:, ::2] = 150
    person = service.create_from_inputs("Alice", [ImageInput.from_camera(frame)])
    sample = repository.list_samples(person.id)[0]
    (settings.data_dir / sample.image_path).write_bytes(b"replaced outside application")

    report = verify_library(repository, settings)

    assert not report.is_valid
    assert report.failures[0].kind == "image_hash_mismatch"
    repository.close()
