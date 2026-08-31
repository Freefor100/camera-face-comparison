from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from camera_face_comparison.config import load_settings
from camera_face_comparison.enrollment import REQUIRED_POSES, EnrollmentService
from camera_face_comparison.face_engine import FaceObservation
from camera_face_comparison.repository import FaceRepository


class FakeFaceEngine:
    def extract_single_face(self, frame: np.ndarray) -> FaceObservation:
        return FaceObservation(
            bbox=(0.0, 0.0, 180.0, 180.0),
            detection_score=0.95,
            embedding=np.array([0.2, 0.3, 0.4], dtype=np.float32),
            blur_variance=150.0,
            landmarks=None,
        )


def _save_image(path: Path, frame: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"test image")


def test_enrollment_commits_person_only_after_five_valid_poses(tmp_path) -> None:
    """An incomplete capture sequence must not create a partially enrolled identity."""

    settings = load_settings(tmp_path)
    repository = FaceRepository(settings.database_path)
    service = EnrollmentService(
        repository=repository,
        settings=settings,
        face_engine=FakeFaceEngine(),
        image_saver=_save_image,
    )
    session = service.begin("Alice")
    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    session.capture(REQUIRED_POSES[0], frame)
    with pytest.raises(ValueError, match="five valid captures"):
        session.commit()

    for pose in REQUIRED_POSES[1:]:
        session.capture(pose, frame)
    person = session.commit()

    assert person.display_name == "Alice"
    assert len(repository.list_people()) == 1
    assert [sample.pose for sample in repository.list_samples()] == list(REQUIRED_POSES)
    assert all((settings.data_dir / sample.image_path).is_file() for sample in repository.list_samples())
    repository.close()


def test_enrollment_rolls_back_when_an_image_cannot_be_saved(tmp_path) -> None:
    """A failed fifth-step write must not leave an identity or face files behind."""

    settings = load_settings(tmp_path)
    repository = FaceRepository(settings.database_path)

    def failing_saver(path: Path, frame: np.ndarray) -> None:
        if path.stem == "left":
            raise OSError("disk write failed")
        _save_image(path, frame)

    service = EnrollmentService(
        repository=repository,
        settings=settings,
        face_engine=FakeFaceEngine(),
        image_saver=failing_saver,
    )
    session = service.begin("Alice")
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    for pose in REQUIRED_POSES:
        session.capture(pose, frame)

    with pytest.raises(OSError, match="disk write failed"):
        session.commit()

    assert repository.list_people() == []
    assert repository.list_samples() == []
    assert list(settings.faces_dir.iterdir()) == []
    repository.close()


def test_existing_person_can_receive_an_extra_valid_sample(tmp_path) -> None:
    settings = load_settings(tmp_path)
    repository = FaceRepository(settings.database_path)
    person = repository.create_person("Alice")
    service = EnrollmentService(
        repository=repository,
        settings=settings,
        face_engine=FakeFaceEngine(),
        image_saver=_save_image,
    )
    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    service.append_sample(person.id, "front", frame)

    samples = repository.list_samples(person.id)
    assert len(samples) == 1
    assert samples[0].pose == "front"
    assert (settings.data_dir / samples[0].image_path).is_file()
    with pytest.raises(ValueError, match="person does not exist"):
        service.append_sample("not-a-person", "front", frame)
    repository.close()
