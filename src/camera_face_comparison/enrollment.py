from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
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


class FaceExtractor(Protocol):
    def extract_single_face(self, frame: np.ndarray) -> FaceObservation: ...


ImageSaver = Callable[[Path, np.ndarray], None]


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

    def create_from_inputs(self, display_name: str, inputs: list[ImageInput]) -> Person:
        """Create one identity from quality-approved camera or local-image inputs.

        The method has no pose sequence. The first usable sample activates the identity.
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
            )
            return len(samples)
        except Exception:
            rmtree(staging_dir, ignore_errors=True)
            for moved_path in moved_paths:
                moved_path.unlink(missing_ok=True)
            raise
        finally:
            _remove_empty_directory(staging_dir.parent)

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
