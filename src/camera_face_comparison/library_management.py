"""协调人员与样本的修改、删除以及受管图片文件清理。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import Settings
from .domain import Person, SampleMetadata
from .repository import FaceRepository


@dataclass(frozen=True)
class LibraryDeletionResult:
    """描述一次标准库删除操作及其磁盘清理结果。"""

    person_id: str
    removed_sample_count: int
    cleanup_failures: tuple[Path, ...]


class LibraryManagementService:
    """协调人员资料修改、数据库删除和受管图片文件清理。"""

    def __init__(self, repository: FaceRepository, settings: Settings) -> None:
        """保存标准库仓库和数据目录配置，不立即修改任何数据。"""

        self._repository = repository
        self._settings = settings
        self._faces_root = settings.faces_dir.resolve()

    def rename_person(self, person_id: str, display_name: str) -> Person:
        """修改人员显示姓名，并返回更新后的人员对象。"""

        return self._repository.rename_person(person_id, display_name)

    def delete_sample(self, person_id: str, sample_id: str) -> LibraryDeletionResult:
        """仓库提交前校验归属与路径，提交后只清理实际删除的元信息。"""
        removed = self._repository.delete_sample(person_id, sample_id)
        return LibraryDeletionResult(removed.person_id, 1, self._cleanup_removed((removed,)))

    def delete_person(self, person_id: str) -> LibraryDeletionResult:
        """先提交删除，再清理事务实际删掉的样本，保留失败路径。"""
        removed = self._repository.delete_person(person_id)
        return LibraryDeletionResult(person_id, len(removed), self._cleanup_removed(removed))

    def _cleanup_removed(self, samples) -> tuple[Path, ...]:
        """提交后的引用检查或清理故障只报告警告，不抹掉已提交事实。"""
        import sqlite3

        removable, failures = [], []
        for sample in samples:
            path = self._settings.data_dir / sample.image_path
            try:
                path = self._managed_image_path(sample)
                if not self._repository.image_is_referenced(path):
                    removable.append(path)
            except (sqlite3.Error, OSError, ValueError, TypeError):
                # 剩余样本的坏字段可能使引用检查失败；留图并报告，不遮蔽已提交的删除。
                failures.append(path)
        return tuple(failures) + self._cleanup_images(tuple(removable))

    def _managed_image_path(self, sample: SampleMetadata) -> Path:
        """把样本路径解析为受管图片路径，并拒绝目录之外的目标。"""

        stored_path = Path(sample.image_path)
        candidate = (
            stored_path.resolve()
            if stored_path.is_absolute()
            else (self._settings.data_dir / stored_path).resolve()
        )
        try:
            # resolve + relative_to 同时阻止 ..、绝对路径和符号链接逃逸受管目录。
            relative = candidate.relative_to(self._faces_root)
        except ValueError as error:
            raise ValueError("样本图片不在受管图片目录中，已停止删除") from error
        if not relative.parts:
            raise ValueError("样本图片路径不能指向受管图片目录本身")
        return candidate

    @staticmethod
    def _cleanup_images(paths: tuple[Path, ...]) -> tuple[Path, ...]:
        """清理数据库已经不再引用的图片，并尝试移除空人员目录。"""

        failures: list[Path] = []
        parent_directories: set[Path] = set()
        for path in paths:
            parent_directories.add(path.parent)
            try:
                path.unlink(missing_ok=True)
            except OSError:
                failures.append(path)
        # 所有图片处理完再删空目录，避免同一人员的多个样本互相影响清理顺序。
        for directory in sorted(parent_directories, key=lambda item: len(item.parts), reverse=True):
            try:
                directory.rmdir()
            except OSError:
                pass
        return tuple(failures)
