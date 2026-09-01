from __future__ import annotations

from pathlib import Path

from camera_face_comparison.__main__ import parse_args


def test_cli_accepts_portable_data_directory() -> None:
    """复制到其他位置的项目必须能够指向自己的本地数据目录。"""

    args = parse_args(["--data-dir", "demo-data"])

    assert args.data_dir == Path("demo-data")
