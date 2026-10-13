"""验证标准库对文件替换、哈希和特征异常的检测。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from camera_face_comparison.config import load_settings
from camera_face_comparison.enrollment import EnrollmentService
from camera_face_comparison.face_engine import FaceObservation
from camera_face_comparison.face_library import InMemoryFaceLibrary
from camera_face_comparison.image_input import ImageInput
from camera_face_comparison.integrity import verify_library
from camera_face_comparison.open_set_policy import ScoreThresholdPolicy
from camera_face_comparison.repository import FaceRepository, SampleInput
from tests.application.helpers import embedding as production_embedding


class FakeFaceEngine:
    """返回固定合格观察结果的完整性测试引擎。"""

    def extract_single_face(self, frame: np.ndarray) -> FaceObservation:
        """为任意输入返回固定的人脸特征和质量指标。"""
        return FaceObservation(
            bbox=(0.0, 0.0, 180.0, 180.0),
            detection_score=0.95,
            embedding=production_embedding([0.2, 0.5, 0.8]),
            blur_variance=150.0,
            landmarks=None,
        )


def _save_image(path: Path, frame: np.ndarray) -> None:
    """保存完整性测试使用的样本文件占位内容。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"valid sample")


@pytest.mark.parametrize("image_problem", ["missing", "replaced"])
def test_photo_problem_warns_without_disabling_valid_embedding(tmp_path, image_problem) -> None:
    """照片缺失或被替换只影响照片展示，已验证的向量仍能识别。"""

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
    photo = settings.data_dir / sample.image_path
    if image_problem == "missing":
        photo.unlink()
    else:
        photo.write_bytes(b"replaced outside application")

    report = verify_library(repository, settings)

    assert report.is_valid
    assert report.warnings
    assert report.metadata[0].image_issue
    snapshot = InMemoryFaceLibrary.from_samples(
        report.recognizable_people, report.samples
    ).snapshot()
    decision = snapshot.search(sample.embedding, ScoreThresholdPolicy(minimum_score=0.5))
    assert decision.accepted_person_id == person.id
    repository.close()


@pytest.mark.parametrize("vector_problem", ["dimension", "hash"])
def test_bad_vector_is_excluded_before_person_average(tmp_path, vector_problem) -> None:
    """坏向量不能进入平均向量，但不影响同一人员的其他有效样本。"""
    settings = load_settings(tmp_path)
    repository = FaceRepository(settings.database_path)
    repository.create_person_with_samples(
        person_id="a",
        display_name="甲",
        samples=[
            SampleInput("faces/a/first.jpg", production_embedding([1, 0]), {}),
            SampleInput("faces/a/second.jpg", production_embedding([0, 1]), {}),
        ],
    )
    first, second = repository.list_samples("a")
    if vector_problem == "dimension":
        repository._connection.execute(
            "UPDATE face_samples SET embedding_dim=511 WHERE id=?", (first.id,)
        )
    else:
        repository._connection.execute(
            "UPDATE face_samples SET embedding_blob=? WHERE id=?",
            (production_embedding([1, 1]).tobytes(), first.id),
        )
    repository._connection.commit()
    report = verify_library(repository, settings)
    assert report.is_valid
    assert [item.id for item in report.samples] == [second.id]
    assert next(item for item in report.metadata if item.id == first.id).recognition_issue
    snapshot = InMemoryFaceLibrary.from_samples(
        report.recognizable_people, report.samples
    ).snapshot()
    np.testing.assert_allclose(snapshot.prototype_matrix[0], production_embedding([0, 1]))
    repository.close()


def test_identity_without_valid_vectors_remains_visible_but_is_not_searched(tmp_path) -> None:
    """失去最后一个有效向量的人员保留在目录中，不拖累其他身份。"""
    settings = load_settings(tmp_path)
    repository = FaceRepository(settings.database_path)
    for person_id, name in (("a", "甲"), ("b", "乙")):
        repository.create_person_with_samples(
            person_id=person_id,
            display_name=name,
            samples=[
                SampleInput(f"faces/{person_id}/sample.jpg", production_embedding([1, 0]), {})
            ],
        )
    repository._connection.execute("UPDATE face_samples SET embedding_dim=511 WHERE person_id='a'")
    repository._connection.commit()
    report = verify_library(repository, settings)
    assert report.is_valid
    assert {person.id for person in report.people} == {"a", "b"}
    snapshot = InMemoryFaceLibrary.from_samples(
        report.recognizable_people, report.samples
    ).snapshot()
    assert snapshot.person_ids == ("b",)
    assert any("甲" in warning and "不参与识别" in warning for warning in report.warnings)
    repository.close()
