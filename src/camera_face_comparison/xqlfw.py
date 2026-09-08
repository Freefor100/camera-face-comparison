from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class XqlfwPair:
    """一条 XQLFW 官方验证对及其所属折次。"""

    left_path: str
    right_path: str
    same_identity: bool
    fold: int


@dataclass(frozen=True)
class XqlfwProtocol:
    """从官方 pairs 文件展开的 XQLFW 验证协议。"""

    fold_count: int
    pairs_per_fold: int
    pairs: tuple[XqlfwPair, ...]
    image_paths: tuple[str, ...]


def load_xqlfw_protocol(pairs_path: Path, dataset_dir: Path) -> XqlfwProtocol:
    """读取 XQLFW 官方 pairs 文件并检查全部图片路径。

    参数：
        pairs_path：XQLFW 发布的 `xqlfw_pairs.txt` 文件。
        dataset_dir：解压后的身份目录根路径。
    返回：
        包含全部折次、正负验证对和去重图片路径的协议对象。
    前置条件：
        pairs 文件使用 LFW 标准格式，第一行是“折次数 每折正样本数”，
        随后每折依次排列正样本和负样本。
    """

    lines = [line.strip() for line in pairs_path.read_text(encoding="utf-8").splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        raise ValueError("XQLFW pairs file is empty")
    try:
        fold_count, pairs_per_fold = (int(value) for value in lines[0].split())
    except (ValueError, TypeError) as error:
        raise ValueError("invalid XQLFW pairs header") from error
    if fold_count < 1 or pairs_per_fold < 1:
        raise ValueError("XQLFW pairs header must contain positive counts")
    expected_rows = fold_count * pairs_per_fold * 2
    if len(lines) - 1 != expected_rows:
        raise ValueError(
            f"XQLFW pairs count mismatch: expected {expected_rows}, got {len(lines) - 1}"
        )

    pairs: list[XqlfwPair] = []
    image_paths: OrderedDict[str, None] = OrderedDict()
    offset = 1
    for fold in range(1, fold_count + 1):
        for same_identity in (True, False):
            for _ in range(pairs_per_fold):
                fields = lines[offset].split()
                offset += 1
                left_path, right_path = _parse_pair(fields, same_identity)
                for relative_path in (left_path, right_path):
                    if not (dataset_dir / relative_path).is_file():
                        raise FileNotFoundError(
                            f"XQLFW image referenced by pairs file is missing: {relative_path}"
                        )
                    image_paths.setdefault(relative_path, None)
                pairs.append(XqlfwPair(left_path, right_path, same_identity, fold))

    return XqlfwProtocol(
        fold_count=fold_count,
        pairs_per_fold=pairs_per_fold,
        pairs=tuple(pairs),
        image_paths=tuple(image_paths),
    )


def _parse_pair(fields: list[str], same_identity: bool) -> tuple[str, str]:
    """把 LFW 格式的一行正样本或负样本转换为相对图片路径。"""

    if same_identity and len(fields) == 3:
        identity, left_index, right_index = fields
        return _image_path(identity, left_index), _image_path(identity, right_index)
    if not same_identity and len(fields) == 4:
        left_identity, left_index, right_identity, right_index = fields
        return _image_path(left_identity, left_index), _image_path(right_identity, right_index)
    kind = "positive" if same_identity else "negative"
    raise ValueError(f"invalid XQLFW {kind} pair row")


def _image_path(identity: str, index: str) -> str:
    """构造 XQLFW/LFW 约定的身份相对图片路径。"""

    try:
        numeric_index = int(index)
    except ValueError as error:
        raise ValueError(f"invalid XQLFW image index: {index}") from error
    if numeric_index < 1:
        raise ValueError("XQLFW image index must be positive")
    return f"{identity}/{identity}_{numeric_index:04d}.jpg"
