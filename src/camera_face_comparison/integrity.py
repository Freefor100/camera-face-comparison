"""用一次数据库一致读取诊断标准库，并提供同源的矩阵和展示数据。"""

import hashlib
from dataclasses import dataclass, replace
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
    """同源目录与有效向量；局部样本问题只警告，结构错误阻断整库。"""

    failures: tuple[IntegrityFailure, ...]
    warnings: tuple[str, ...] = ()
    people: tuple[Person, ...] = ()
    samples: tuple[FaceSample, ...] = ()
    metadata: tuple[SampleMetadata, ...] = ()

    @property
    def is_valid(self) -> bool:
        """返回是否允许使用已验证样本，不要求所有照片和人员均可识别。"""
        return not self.failures

    @property
    def recognizable_people(self) -> tuple[Person, ...]:
        """返回仍有有效向量的人员，完整人员目录保留用于管理。"""
        person_ids = {sample.person_id for sample in self.samples}
        return tuple(person for person in self.people if person.id in person_ids)


def verify_library(repository: FaceRepository, settings: Settings) -> LibraryVerificationReport:
    """一次读取后分开检查照片和向量，只把有效向量交给识别。"""
    people_rows, sample_rows, checks, foreign_keys = repository.read_library_rows()
    failures = [
        IntegrityFailure("sqlite_integrity_error", value)
        for value in checks
        if value.lower() != "ok"
    ]
    failures.extend(IntegrityFailure("foreign_key_violation", value) for value in foreign_keys)
    people, samples, metadata, warnings = [], [], [], []
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
            item = repository._metadata_from_row(row)
            path = (settings.data_dir / item.image_path).resolve()
            if not path.is_relative_to(root) or path == root:
                failures.append(
                    IntegrityFailure("image_path_outside_library", str(path), sample_id)
                )
                metadata.append(replace(item, image_issue="照片路径不在标准库目录内。"))
                continue
            if path in paths:
                failures.append(IntegrityFailure("shared_image_path", str(path), sample_id))
            paths.add(path)
        except (ValueError, TypeError, OSError) as error:
            # 归属或路径不可恢复时不能安全管理文件，仍按整库结构异常处理。
            failures.append(IntegrityFailure("invalid_sample_metadata", str(error), sample_id))
            continue

        image_issue = _check_image(path, row["image_sha256"])
        if image_issue:
            warnings.append(f"样本 {sample_id}：{image_issue} 路径：{item.image_path}")
        recognition_issue = None
        try:
            sample = repository._sample_from_row(row)
            digest = hashlib.sha256(sample.embedding.astype(np.float32).tobytes()).hexdigest()
            if sample.embedding_sha256 != digest:
                raise ValueError("向量哈希与登记值不一致")
            # 哈希和向量契约全部通过后才纳入平均向量；坏样本不能污染正常样本。
            samples.append(sample)
            grouped.setdefault(sample.person_id, []).append(sample.embedding)
        except (ValueError, TypeError, AttributeError, OverflowError, OSError) as error:
            recognition_issue = f"样本特征数据异常，已排除识别：{error}"
            warnings.append(f"样本 {sample_id}：{recognition_issue}")
        metadata.append(replace(item, image_issue=image_issue, recognition_issue=recognition_issue))
    excluded_people = set()
    for person in people:
        vectors = grouped.get(person.id, [])
        if not vectors:
            warnings.append(
                f"“{person.display_name}”没有有效特征向量，不参与识别；可追加有效样本。"
            )
        else:
            try:
                normalize_embedding(np.mean([normalize_embedding(v) for v in vectors], axis=0))
            except ValueError as error:
                excluded_people.add(person.id)
                warnings.append(f"“{person.display_name}”无法形成有效平均向量，不参与识别：{error}")
    if excluded_people:
        samples = [sample for sample in samples if sample.person_id not in excluded_people]
        metadata = [
            replace(item, recognition_issue="人员平均向量无效，不参与识别。")
            if item.person_id in excluded_people
            else item
            for item in metadata
        ]
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


def _check_image(path: Path, expected_hash: str | None) -> str | None:
    """检查受管照片的文件字节；照片问题不改变已保存向量，也不自动修复。"""
    try:
        if not path.is_file():
            return "照片缺失或不是普通文件；有效向量仍可使用。"
        if expected_hash is None:
            return "照片缺少登记哈希，无法确认原文件；有效向量仍可使用。"
        if _sha256_file(path) != expected_hash:
            return "照片内容发生变化，不再展示为原登记照片；有效向量仍可使用。"
    except OSError as error:
        return f"照片无法读取；有效向量仍可使用。原因：{error}"
    return None


def _sha256_file(path: Path) -> str:
    """分块读取文件并计算哈希，不把整张图片加载进内存。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()
