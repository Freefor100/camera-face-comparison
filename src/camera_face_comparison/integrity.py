"""用一次数据库一致读取诊断标准库，并提供同源的矩阵和展示数据。"""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import Settings
from .domain import FaceSample, Person, SampleMetadata
from .face_engine import normalize_embedding
from .repository import FaceRepository


@dataclass(frozen=True)
class IntegrityFailure:
    """一个阻止标准库参与识别的异常，保留具体样本和原因。"""

    kind: str
    detail: str
    sample_id: str | None = None


@dataclass(frozen=True)
class LibraryVerificationReport:
    """检查结果及来自同一数据库快照的人员、样本和展示元信息。"""

    failures: tuple[IntegrityFailure, ...]
    warnings: tuple[str, ...] = ()
    people: tuple[Person, ...] = ()
    samples: tuple[FaceSample, ...] = ()
    metadata: tuple[SampleMetadata, ...] = ()

    @property
    def is_valid(self) -> bool:
        """返回能否安全构建所有人员原型，清理警告不阻断识别。"""
        return not self.failures


def verify_library(repository: FaceRepository, settings: Settings) -> LibraryVerificationReport:
    """一次读取结构和记录，再逐条汇总向量、文件和身份不变量。"""
    people_rows, sample_rows, checks, foreign_keys = repository.read_library_rows()
    failures = [
        IntegrityFailure("sqlite_integrity_error", value)
        for value in checks
        if value.lower() != "ok"
    ]
    failures.extend(IntegrityFailure("foreign_key_violation", value) for value in foreign_keys)
    people, samples, metadata = [], [], []
    for row in people_rows:
        try:
            people.append(repository._person_from_row(row))
        except (ValueError, TypeError) as error:
            failures.append(IntegrityFailure("invalid_person", f"{row['id']}：{error}"))
    root = settings.faces_dir.resolve()
    paths: set[Path] = set()
    grouped = {}
    for row in sample_rows:
        sample_id = str(row["id"])
        try:
            metadata.append(repository._metadata_from_row(row))
            sample = repository._sample_from_row(row)
            samples.append(sample)
            grouped.setdefault(sample.person_id, []).append(sample.embedding)
            path = (settings.data_dir / sample.image_path).resolve()
            if not path.is_relative_to(root) or path == root:
                failures.append(
                    IntegrityFailure("image_path_outside_library", str(path), sample_id)
                )
                continue
            if path in paths:
                failures.append(IntegrityFailure("shared_image_path", str(path), sample_id))
            paths.add(path)
            if not path.is_file():
                failures.append(IntegrityFailure("image_missing", str(path), sample_id))
            elif sample.image_sha256 is None:
                failures.append(IntegrityFailure("missing_image_hash", str(path), sample_id))
            elif _sha256_file(path) != sample.image_sha256:
                failures.append(IntegrityFailure("image_hash_mismatch", str(path), sample_id))
            digest = hashlib.sha256(sample.embedding.astype(np.float32).tobytes()).hexdigest()
            if sample.embedding_sha256 != digest:
                failures.append(IntegrityFailure("embedding_hash_mismatch", sample_id, sample_id))
        except (ValueError, TypeError, AttributeError, OverflowError, OSError) as error:
            failures.append(IntegrityFailure("invalid_sample", str(error), sample_id))
    for person in people:
        vectors = grouped.get(person.id, [])
        if not vectors:
            failures.append(IntegrityFailure("person_without_valid_samples", person.display_name))
        else:
            try:
                normalize_embedding(np.mean([normalize_embedding(v) for v in vectors], axis=0))
            except ValueError as error:
                failures.append(IntegrityFailure("invalid_person_prototype", str(error)))
    warnings = []
    try:
        # 只扫描受管目录，遗留文件不自动清理，也不误报成识别数据损坏。
        for path in root.rglob("*"):
            if path.is_file() and path.resolve() not in paths:
                warnings.append(f"未被数据库引用的文件：{path.relative_to(root)}")
    except OSError as error:
        warnings.append(f"遗留文件扫描未完成：{error}")
    return LibraryVerificationReport(
        tuple(failures), tuple(warnings), tuple(people), tuple(samples), tuple(metadata)
    )


def _sha256_file(path: Path) -> str:
    """分块读取文件并计算哈希，不把整张图片加载进内存。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()
