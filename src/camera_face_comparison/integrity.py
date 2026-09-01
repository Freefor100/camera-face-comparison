from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import Settings
from .repository import FaceRepository


@dataclass(frozen=True)
class IntegrityFailure:
    """一个不应继续用于识别的本地标准库异常项。"""

    kind: str
    detail: str
    sample_id: str | None = None


@dataclass(frozen=True)
class LibraryVerificationReport:
    """当前 SQLite 数据库和人脸文件的只读完整性检查结果。"""

    failures: tuple[IntegrityFailure, ...]

    @property
    def is_valid(self) -> bool:
        """返回是否不存在完整性失败项。"""
        return not self.failures


def verify_library(repository: FaceRepository, settings: Settings) -> LibraryVerificationReport:
    """检查 SQLite 结构以及录入时记录的图片、特征哈希。

    参数：
        repository：要检查的本地人脸库仓库。
        settings：用于解析相对图片路径的数据目录配置。
    返回：
        所有发现的异常；没有异常时报告为有效。
    前置条件：
        仓库连接已打开，样本图片路径应位于配置的数据目录内。
    """

    failures: list[IntegrityFailure] = []
    for message in repository.sqlite_integrity_messages():
        if message.lower() != "ok":
            failures.append(IntegrityFailure(kind="sqlite_integrity_error", detail=message))
    for violation in repository.foreign_key_violations():
        failures.append(IntegrityFailure(kind="foreign_key_violation", detail=violation))

    face_root = settings.faces_dir.resolve()
    for sample in repository.list_samples():
        image_path = (settings.data_dir / sample.image_path).resolve()
        if not _is_within(image_path, face_root):
            failures.append(
                IntegrityFailure("image_path_outside_library", str(image_path), sample.id)
            )
        elif not image_path.is_file():
            failures.append(IntegrityFailure("image_missing", str(image_path), sample.id))
        elif sample.image_sha256 is None:
            failures.append(IntegrityFailure("missing_image_hash", str(image_path), sample.id))
        elif _sha256_file(image_path) != sample.image_sha256:
            failures.append(IntegrityFailure("image_hash_mismatch", str(image_path), sample.id))

        expected_embedding_hash = sample.embedding_sha256
        current_embedding_hash = hashlib.sha256(
            np.asarray(sample.embedding, dtype=np.float32).tobytes()
        ).hexdigest()
        if expected_embedding_hash is None:
            failures.append(IntegrityFailure("missing_embedding_hash", sample.id, sample.id))
        elif current_embedding_hash != expected_embedding_hash:
            failures.append(IntegrityFailure("embedding_hash_mismatch", sample.id, sample.id))
    return LibraryVerificationReport(tuple(failures))


def _is_within(path: Path, root: Path) -> bool:
    """判断解析后的路径是否仍位于指定根目录内。"""
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _sha256_file(path: Path) -> str:
    """分块计算本地文件的 SHA-256 哈希。"""
    digest = hashlib.sha256()
    with path.open("rb") as image_file:
        for chunk in iter(lambda: image_file.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()
