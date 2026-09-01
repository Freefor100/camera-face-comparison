from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。

    参数：
        argv：可选的参数序列；省略时读取进程命令行。
    返回：
        包含数据目录路径的参数对象。
    """
    parser = argparse.ArgumentParser(description="Offline camera face comparison application")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path.cwd() / "data",
        help="portable directory for models, the SQLite library, samples, and logs",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """解析参数并启动桌面应用，返回 Qt 退出码。"""
    args = parse_args(argv)
    from .app import run_application

    return run_application(args.data_dir)


if __name__ == "__main__":
    raise SystemExit(main())
