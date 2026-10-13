"""验证标准库重命名、删除和受管图片清理边界。"""

from __future__ import annotations

import hashlib

import pytest

from camera_face_comparison.config import load_settings
from camera_face_comparison.library_management import LibraryManagementService
from camera_face_comparison.repository import FaceRepository, SampleInput
from tests.application.helpers import embedding as production_embedding


def _add_person_with_files(
    repository: FaceRepository,
    data_dir,
    *,
    person_id: str = "alice-id",
    filenames: tuple[str, ...] = ("first.jpg", "second.jpg"),
) -> list:
    """创建真实受管图片及其数据库样本，供删除行为测试使用。"""

    person_dir = data_dir / "faces" / person_id
    person_dir.mkdir(parents=True, exist_ok=True)
    inputs = []
    paths = []
    for index, filename in enumerate(filenames, start=1):
        path = person_dir / filename
        path.write_bytes(f"sample-{index}".encode())
        paths.append(path)
        inputs.append(
            SampleInput(
                image_path=f"faces/{person_id}/{filename}",
                embedding=production_embedding([1.0, float(index)]),
                quality_metrics={},
                source_type="file",
                image_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        )
    repository.create_person_with_samples(
        person_id=person_id,
        display_name="Alice",
        samples=inputs,
    )
    return paths


def test_management_service_deletes_sample_row_and_managed_image(tmp_path) -> None:
    """删除单张样本必须同时移除数据库记录和对应受管图片。"""

    settings = load_settings(tmp_path)
    repository = FaceRepository(settings.database_path)
    paths = _add_person_with_files(repository, settings.data_dir)
    first_sample = repository.list_samples("alice-id")[0]
    service = LibraryManagementService(repository, settings)

    result = service.delete_sample("alice-id", first_sample.id)

    assert result.removed_sample_count == 1
    assert result.cleanup_failures == ()
    assert not paths[0].exists()
    assert paths[1].exists()
    assert len(repository.list_samples("alice-id")) == 1
    repository.close()


def test_management_service_deletes_person_and_managed_directory(tmp_path) -> None:
    """删除人员必须移除全部样本图片，并在目录为空后清理人员目录。"""

    settings = load_settings(tmp_path)
    repository = FaceRepository(settings.database_path)
    paths = _add_person_with_files(repository, settings.data_dir)
    service = LibraryManagementService(repository, settings)

    result = service.delete_person("alice-id")

    assert result.removed_sample_count == 2
    assert result.cleanup_failures == ()
    assert not any(path.exists() for path in paths)
    assert not paths[0].parent.exists()
    assert repository.get_person("alice-id") is None
    repository.close()


def test_management_service_rejects_sample_path_outside_faces_directory(tmp_path) -> None:
    """数据库路径异常时必须先停止删除，不能触碰受管目录之外的文件。"""

    settings = load_settings(tmp_path)
    outside_path = tmp_path / "outside.jpg"
    outside_path.write_bytes(b"outside")
    repository = FaceRepository(settings.database_path)
    repository.create_person_with_samples(
        person_id="alice-id",
        display_name="Alice",
        samples=[
            SampleInput(
                image_path="faces/alice/imported.jpg",
                embedding=production_embedding([1.0, 0.0]),
                quality_metrics={},
                source_type="file",
                image_sha256=hashlib.sha256(outside_path.read_bytes()).hexdigest(),
            )
        ],
    )
    service = LibraryManagementService(repository, settings)

    repository._connection.execute("UPDATE face_samples SET image_path = ?", (str(outside_path),))
    repository._connection.commit()
    with pytest.raises(ValueError, match="受管图片目录"):
        service.delete_person("alice-id")

    assert outside_path.exists()
    assert repository.get_person("alice-id") is not None
    assert len(repository.list_samples("alice-id")) == 1
    repository.close()
