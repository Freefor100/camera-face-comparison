from __future__ import annotations

from pathlib import Path

from camera_face_comparison.__main__ import parse_args


def test_cli_accepts_portable_data_directory() -> None:
    """A copied project must be able to point at its local data folder."""

    args = parse_args(["--data-dir", "demo-data"])

    assert args.data_dir == Path("demo-data")
