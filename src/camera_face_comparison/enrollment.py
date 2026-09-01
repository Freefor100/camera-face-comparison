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
    """提供单人脸检测和特征提取能力的接口。"""

    def extract_single_face(self, frame: np.ndarray) -> FaceObservation:
        """从 BGR 图像中提取一张通过质量门控的人脸。"""
        ...


ImageSaver = Callable[[Path, np.ndarray], None]


class EnrollmentService:
    """协调质量检查、图片保存和标准人脸库持久化。"""

    def __init__(
        self,
        *,
        repository: FaceRepository,
        settings: Settings,
        face_engine: FaceExtractor,
        image_saver: ImageSaver,
    ) -> None:
        """保存录入服务依赖，不在构造阶段执行人脸推理。"""
        self.repository = repository
        self.settings = settings
        self.face_engine = face_engine
        self.image_saver = image_saver

    def create_from_inputs(self, display_name: str, inputs: list[ImageInput]) -> Person:
        """从摄像头或本地图片创建一个新身份。

        参数：
            display_name：要显示的人员姓名。
            inputs：摄像头或本地图片输入，至少包含一项。
        返回：
            新建的人员对象；第一张合格样本写入后即可参与识别。
        前置条件：
            所有输入都必须恰好检测到一张符合当前质量规则的人脸；任一项失败都会回滚。
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
        """向已有身份追加摄像头或本地图片样本。

        参数：
            person_id：已存在身份的编号。
            inputs：待追加的图片输入，至少包含一项。
        返回：
            实际追加的合格样本数量。
        前置条件：
            所有输入都必须通过单人脸和质量检查；任一项失败都会回滚本次追加。
        """

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
    """分块计算图片文件的 SHA-256 哈希。"""
    digest = hashlib.sha256()
    with path.open("rb") as image_file:
        for chunk in iter(lambda: image_file.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _remove_empty_directory(path: Path) -> None:
    """仅在目录为空时清理临时目录，避免影响其他并发操作。"""
    try:
        path.rmdir()
    except OSError:
        pass
