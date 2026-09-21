"""协调人员与样本的修改、删除以及受管图片文件清理。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import Settings
from .domain import FaceSample, Person
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
        """删除一张非末尾样本，并清理其受管图片文件。

        数据库是标准库事实来源，因此数据库事务先提交；文件清理失败只会留下无引用图片，
        不会让识别流程继续引用已经不存在的数据。
        """

        samples = self._repository.list_samples(person_id)
        sample = next((item for item in samples if item.id == sample_id), None)
        if sample is None:
            raise ValueError("样本不存在或不属于所选人员")
        # 删除前先解析并验证路径；数据库提交后只清理已经不再被引用的文件。
        managed_path = self._managed_image_path(sample)
        removed = self._repository.delete_sample(person_id, sample_id)
        failures = self._cleanup_images((managed_path,))
        return LibraryDeletionResult(
            person_id=removed.person_id,
            removed_sample_count=1,
            cleanup_failures=failures,
        )

    def delete_person(self, person_id: str) -> LibraryDeletionResult:
        """删除人员及其全部样本，并清理相应的受管图片文件。"""

        samples = self._repository.list_samples(person_id)
        managed_paths = tuple(self._managed_image_path(sample) for sample in samples)
        removed = self._repository.delete_person(person_id)
        failures = self._cleanup_images(managed_paths)
        return LibraryDeletionResult(
            person_id=person_id,
            removed_sample_count=len(removed),
            cleanup_failures=failures,
        )

    def _managed_image_path(self, sample: FaceSample) -> Path:
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
