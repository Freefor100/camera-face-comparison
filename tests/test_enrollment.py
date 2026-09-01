from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from camera_face_comparison.config import load_settings
from camera_face_comparison.enrollment import REQUIRED_POSES, EnrollmentService
from camera_face_comparison.face_engine import FaceObservation
from camera_face_comparison.image_input import ImageInput
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


class DiverseFakeFaceEngine:
    def extract_single_face(self, frame: np.ndarray) -> FaceObservation:
        marker = float(frame[0, 0, 0])
        return FaceObservation(
            bbox=(0.0, 0.0, 180.0, 180.0),
            detection_score=0.95,
            embedding=np.array([marker + 1.0, 3.0, 7.0], dtype=np.float32),
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


def test_three_valid_local_inputs_activate_a_person_without_pose_steps(tmp_path) -> None:
    """Local images can create an active person without a prescribed capture sequence."""

    settings = load_settings(tmp_path)
    repository = FaceRepository(settings.database_path)
    service = EnrollmentService(
        repository=repository,
        settings=settings,
        face_engine=DiverseFakeFaceEngine(),
        image_saver=_save_image,
    )
    inputs = []
    for value in (80, 120, 180):
        frame = np.full((240, 320, 3), value, dtype=np.uint8)
        frame[:, ::2] = value + 30
        inputs.append(ImageInput.from_camera(frame))

    person = service.create_from_inputs("Alice", inputs)

    restored = repository.get_person(person.id)
    samples = repository.list_samples(person.id)
    assert restored is not None
    assert restored.lifecycle == "active"
    assert len(samples) == 3
    assert all(sample.source_type == "camera" for sample in samples)
    assert all(sample.image_sha256 for sample in samples)
    assert all((settings.data_dir / sample.image_path).is_file() for sample in samples)
    repository.close()


def test_local_inputs_can_activate_an_existing_draft_person(tmp_path) -> None:
    """Imported images must expand a draft identity without reopening a pose-by-pose workflow."""

    settings = load_settings(tmp_path)
    repository = FaceRepository(settings.database_path)
    service = EnrollmentService(
        repository=repository,
        settings=settings,
        face_engine=DiverseFakeFaceEngine(),
        image_saver=_save_image,
    )
    person = service.create_from_inputs("Alice", [ImageInput.from_camera(_textured_frame(80))])

    service.append_from_inputs(
        person.id,
        [
            ImageInput.from_camera(_textured_frame(120)),
            ImageInput.from_camera(_textured_frame(180)),
        ],
    )

    restored = repository.get_person(person.id)
    samples = repository.list_samples(person.id)
    assert restored is not None
    assert restored.lifecycle == "active"
    assert len(samples) == 3
    repository.close()


def _textured_frame(value: int) -> np.ndarray:
    frame = np.full((240, 320, 3), value, dtype=np.uint8)
    frame[:, ::2] = value + 30
    return frame
