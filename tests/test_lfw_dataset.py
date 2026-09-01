from __future__ import annotations

import json

import pytest

from camera_face_comparison.lfw_dataset import build_lfw_protocol, write_lfw_protocol


def test_lfw_protocol_keeps_known_and_unknown_people_disjoint(tmp_path) -> None:
    """可复现的开放集划分不能把未知身份同时放入标准库。"""

    dataset_dir = tmp_path / "lfw-deepfunneled"
    for name in ("Alice", "Bob", "Carol", "Dave", "Eve"):
        person_dir = dataset_dir / name
        person_dir.mkdir(parents=True)
        for index in range(1, 8):
            (person_dir / f"{name}_{index:04d}.jpg").write_bytes(b"fixture")

    protocol = build_lfw_protocol(
        dataset_dir,
        known_identity_count=3,
        unknown_identity_count=2,
        enrollment_per_identity=5,
        probes_per_identity=2,
    )

    assert set(protocol.enrollment) == {"Alice", "Bob", "Carol"}
    assert len(protocol.probes) == 10
    assert {probe.expected_person_id for probe in protocol.probes[:6]} == {"Alice", "Bob", "Carol"}
    assert {probe.expected_person_id for probe in protocol.probes[6:]} == {None}

    output = tmp_path / "lfw_protocol.json"
    write_lfw_protocol(protocol, output)
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["protocol"] == "lfw-open-set-v1"
    assert saved["enrollment"]["Alice"][0] == "Alice/Alice_0001.jpg"


def test_lfw_downloader_uses_the_get_accessible_mirror_and_reports_network_errors(tmp_path) -> None:
    """公共数据下载失败时必须指出可用来源并给出清晰的恢复提示。"""

    requested_urls: list[str] = []

    def unavailable_downloader(url: str, destination: str) -> None:
        """记录下载地址并模拟网络中断。"""
        requested_urls.append(url)
        raise OSError("network interrupted")

    from camera_face_comparison.lfw_dataset import ensure_lfw_dataset

    with pytest.raises(RuntimeError, match="could not download LFW"):
        ensure_lfw_dataset(tmp_path, download=True, downloader=unavailable_downloader)

    assert requested_urls == ["https://ndownloader.figshare.com/files/5976015"]


def test_lfw_loader_rejects_an_unverified_archive_without_extracting_it(tmp_path) -> None:
    """不完整压缩包不能被当作有效的本地数据集解压使用。"""

    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()
    (datasets_dir / "lfw-funneled.tgz").write_bytes(b"truncated")

    from camera_face_comparison.lfw_dataset import ensure_lfw_dataset

    with pytest.raises(RuntimeError, match="checksum mismatch"):
        ensure_lfw_dataset(tmp_path, download=False)
