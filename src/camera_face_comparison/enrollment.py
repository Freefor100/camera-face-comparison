from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from shutil import rmtree
from typing import Protocol
from uuid import uuid4

import numpy as np

from .config import Settings
from .domain import Person
from .face_engine import FaceInputError, FaceObservation, validate_single_face
from .image_input import ImageInput, assess_quality
from .repository import FaceRepository, SampleInput

REQUIRED_POSES = ("front", "left", "right", "up", "down")


class FaceExtractor(Protocol):
    def extract_single_face(self, frame: np.ndarray) -> FaceObservation: ...


ImageSaver = Callable[[Path, np.ndarray], None]


@dataclass
class EnrollmentSession:
    """An in-memory five-pose collection that is committed only when complete."""

    _service: EnrollmentService
    display_name: str
    _captures: dict[str, tuple[np.ndarray, FaceObservation]] = field(default_factory=dict)

    @property
    def completed_poses(self) -> tuple[str, ...]:
        return tuple(pose for pose in REQUIRED_POSES if pose in self._captures)

    def capture(self, pose: str, frame: np.ndarray) -> FaceObservation:
        if pose not in REQUIRED_POSES:
            raise ValueError(f"unsupported pose: {pose}")
        if pose in self._captures:
            raise ValueError(f"pose already captured: {pose}")
        observation = self._service.face_engine.extract_single_face(frame)
        validated = validate_single_face([observation], self._service.settings)
        self._captures[pose] = (frame.copy(), validated)
        return validated

    def commit(self) -> Person:
        missing = [pose for pose in REQUIRED_POSES if pose not in self._captures]
        if missing:
            raise ValueError("five valid captures are required before enrollment")
        return self._service._commit(self.display_name, self._captures)


class EnrollmentService:
    """Coordinates quality checks, image saving, and durable face-library updates."""

    def __init__(
        self,
        *,
        repository: FaceRepository,
        settings: Settings,
        face_engine: FaceExtractor,
        image_saver: ImageSaver,
    ) -> None:
        self.repository = repository
        self.settings = settings
        self.face_engine = face_engine
        self.image_saver = image_saver

    def begin(self, display_name: str) -> EnrollmentSession:
        if not display_name.strip():
            raise ValueError("display_name must not be empty")
        return EnrollmentSession(self, display_name.strip())

    def append_sample(
        self,
        person_id: str,
        pose: str,
        frame: np.ndarray,
    ) -> FaceObservation:
        """Append one quality-approved sample to an already enrolled person."""

        if pose not in REQUIRED_POSES:
            raise ValueError(f"unsupported pose: {pose}")
        if self.repository.get_person(person_id) is None:
            raise ValueError("person does not exist")
        observation = self.face_engine.extract_single_face(frame)
        validated = validate_single_face([observation], self.settings)
        image_path = self.settings.faces_dir / person_id / f"{pose}_extra_{uuid4().hex}.jpg"
        self.image_saver(image_path, frame)
        try:
            relative_path = image_path.relative_to(self.settings.data_dir).as_posix()
            self.repository.add_sample(
                person_id=person_id,
                image_path=relative_path,
                embedding=validated.embedding,
                pose=pose,
                quality={
                    "detection_score": validated.detection_score,
                    "blur_variance": validated.blur_variance,
                },
                image_sha256=_sha256_file(image_path),
                min_active_samples=self.settings.min_active_samples,
            )
        except Exception:
            image_path.unlink(missing_ok=True)
            raise
        return validated

    def create_from_inputs(self, display_name: str, inputs: list[ImageInput]) -> Person:
        """Create one identity from quality-approved camera or local-image inputs.

        The method intentionally has no pose requirement.  A person becomes active only
        after the configured number of usable samples has been stored.
        """

        normalized_name = display_name.strip()
        if not normalized_name:
            raise ValueError("display_name must not be empty")
        if not inputs:
            raise ValueError("at least one image input is required")

        person_id = str(uuid4())
        staging_dir = self.settings.faces_dir / ".staging" / person_id
        final_dir = self.settings.faces_dir / person_id
        samples: list[SampleInput] = []
        try:
            for index, image_input in enumerate(inputs, start=1):
                observation = self.face_engine.extract_single_face(image_input.frame)
                validated = validate_single_face([observation], self.settings)
                profile = assess_quality(image_input.frame, validated, self.settings)
                if profile.tier == "reject":
                    raise FaceInputError(
                        "quality_rejected:" + ",".join(profile.reasons or ("low_score",))
                    )
                staged_path = staging_dir / f"sample_{index:03d}.jpg"
                self.image_saver(staged_path, image_input.frame)
                samples.append(
                    SampleInput(
                        image_path=(Path("faces") / person_id / staged_path.name).as_posix(),
                        embedding=validated.embedding,
                        pose=f"sample_{index:03d}",
                        quality={
                            **profile.metrics,
                            "quality_score": profile.score,
                            "tier": profile.tier,
                        },
                        source_type=image_input.source_type,
                        image_sha256=_sha256_file(staged_path),
                    )
                )
            final_dir.parent.mkdir(parents=True, exist_ok=True)
            staging_dir.replace(final_dir)
            return self.repository.create_person_with_samples(
                person_id=person_id,
                display_name=normalized_name,
                samples=samples,
                min_active_samples=self.settings.min_active_samples,
            )
        except Exception:
            rmtree(staging_dir, ignore_errors=True)
            rmtree(final_dir, ignore_errors=True)
            raise
        finally:
            _remove_empty_directory(staging_dir.parent)

    def append_from_inputs(self, person_id: str, inputs: Sequence[ImageInput]) -> int:
        """Append camera or local-image samples without imposing a pose sequence."""

        if self.repository.get_person(person_id) is None:
            raise ValueError("person does not exist")
        if not inputs:
            raise ValueError("at least one image input is required")

        operation_id = uuid4().hex
        staging_dir = self.settings.faces_dir / ".staging" / f"{person_id}-{operation_id}"
        person_dir = self.settings.faces_dir / person_id
        samples: list[SampleInput] = []
        staged_moves: list[tuple[Path, Path]] = []
        moved_paths: list[Path] = []
        try:
            for index, image_input in enumerate(inputs, start=1):
                observation = self.face_engine.extract_single_face(image_input.frame)
                validated = validate_single_face([observation], self.settings)
                profile = assess_quality(image_input.frame, validated, self.settings)
                if profile.tier == "reject":
                    raise FaceInputError(
                        "quality_rejected:" + ",".join(profile.reasons or ("low_score",))
                    )
                filename = f"sample_{operation_id}_{index:03d}.jpg"
                staged_path = staging_dir / filename
                final_path = person_dir / filename
                self.image_saver(staged_path, image_input.frame)
                staged_moves.append((staged_path, final_path))
                samples.append(
                    SampleInput(
                        image_path=(Path("faces") / person_id / filename).as_posix(),
                        embedding=validated.embedding,
                        pose=f"sample_{index:03d}",
                        quality={
                            **profile.metrics,
                            "quality_score": profile.score,
                            "tier": profile.tier,
                        },
                        source_type=image_input.source_type,
                        image_sha256=_sha256_file(staged_path),
                    )
                )
            person_dir.mkdir(parents=True, exist_ok=True)
            for staged_path, final_path in staged_moves:
                staged_path.replace(final_path)
                moved_paths.append(final_path)
            self.repository.add_samples(
                person_id=person_id,
                samples=samples,
                min_active_samples=self.settings.min_active_samples,
            )
            return len(samples)
        except Exception:
            rmtree(staging_dir, ignore_errors=True)
            for moved_path in moved_paths:
                moved_path.unlink(missing_ok=True)
            raise
        finally:
            _remove_empty_directory(staging_dir.parent)

    def _commit(
        self,
        display_name: str,
        captures: dict[str, tuple[np.ndarray, FaceObservation]],
    ) -> Person:
        person_id = str(uuid4())
        person_dir = self.settings.faces_dir / person_id
        samples: list[SampleInput] = []
        try:
            for pose in REQUIRED_POSES:
                frame, observation = captures[pose]
                image_path = person_dir / f"{pose}.jpg"
                self.image_saver(image_path, frame)
                relative_path = image_path.relative_to(self.settings.data_dir).as_posix()
                samples.append(
                    SampleInput(
                        image_path=relative_path,
                        embedding=observation.embedding,
                        pose=pose,
                        quality={
                            "detection_score": observation.detection_score,
                            "blur_variance": observation.blur_variance,
                        },
                        image_sha256=_sha256_file(image_path),
                    )
                )
            return self.repository.create_person_with_samples(
                person_id=person_id,
                display_name=display_name,
                samples=samples,
                min_active_samples=self.settings.min_active_samples,
            )
        except Exception:
            rmtree(person_dir, ignore_errors=True)
            raise


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as image_file:
        for chunk in iter(lambda: image_file.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _remove_empty_directory(path: Path) -> None:
    try:
        path.rmdir()
    except OSError:
        pass
