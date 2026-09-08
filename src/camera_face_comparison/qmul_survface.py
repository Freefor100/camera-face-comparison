from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .lfw_dataset import LfwProbe, LfwProtocol


def build_qmul_protocol(dataset_root: Path) -> LfwProtocol:
    """按 QMUL-SurvFace 官方开放集识别协议建立 Gallery/Probe 划分。

    参数：
        dataset_root：解压后的 `QMUL-SurvFace` 根目录。
    返回：
        可复用本项目 1:N 评测器的开放集协议；mated probe 带身份标签，
        unmated probe 使用 `None` 表示库外身份。
    前置条件：
        `Face_Identification_Test_Set` 下存在 gallery、mated_probe、
        unmated_probe 和官方 MAT 标签文件；MAT 文件中的图片必须能在磁盘找到。
    """

    test_root = dataset_root / "Face_Identification_Test_Set"
    gallery_dir = test_root / "gallery"
    mated_dir = test_root / "mated_probe"
    unmated_dir = test_root / "unmated_probe"
    required = (gallery_dir, mated_dir, unmated_dir)
    if any(not path.is_dir() for path in required):
        raise FileNotFoundError("QMUL identification test-set directories are incomplete")

    gallery_mat = _load_mat(test_root / "gallery_img_ID_pairs.mat")
    mated_mat = _load_mat(test_root / "mated_probe_img_ID_pairs.mat")
    gallery_ids = _read_strings(gallery_mat, "gallery_ids")
    gallery_names = _read_strings(gallery_mat, "gallery_set")
    mated_ids = _read_strings(mated_mat, "mated_probe_ids")
    mated_names = _read_strings(mated_mat, "mated_probe_set")
    if len(gallery_ids) != len(gallery_names) or len(mated_ids) != len(mated_names):
        raise ValueError("QMUL MAT identity and filename lengths do not match")

    enrollment: dict[str, list[str]] = defaultdict(list)
    for person_id, filename in zip(gallery_ids, gallery_names, strict=True):
        relative_path = f"gallery/{filename}"
        _require_file(test_root / relative_path, relative_path)
        enrollment[person_id].append(relative_path)
    if not enrollment:
        raise ValueError("QMUL protocol contains no Gallery identities")

    probes: list[LfwProbe] = []
    gallery_ids_set = set(enrollment)
    for person_id, filename in zip(mated_ids, mated_names, strict=True):
        if person_id not in gallery_ids_set:
            raise ValueError(f"QMUL mated probe refers to missing Gallery identity: {person_id}")
        relative_path = f"mated_probe/{filename}"
        _require_file(test_root / relative_path, relative_path)
        probes.append(LfwProbe(relative_path, person_id))
    for path in sorted(unmated_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            probes.append(LfwProbe(f"unmated_probe/{path.name}", None))

    return LfwProtocol(
        enrollment={person_id: tuple(paths) for person_id, paths in enrollment.items()},
        probes=tuple(probes),
    )


def write_qmul_protocol(protocol: LfwProtocol, output_path: Path) -> None:
    """把 QMUL 开放集协议写为可回放 JSON，不复制原始图片。"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": "qmul-survface-open-set-v1",
        "enrollment": {person_id: list(paths) for person_id, paths in protocol.enrollment.items()},
        "probes": [
            {
                "relative_path": probe.relative_path,
                "expected_person_id": probe.expected_person_id,
            }
            for probe in protocol.probes
        ],
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_mat(path: Path) -> dict[str, Any]:
    """通过可选 SciPy 读取 QMUL 发布的 MATLAB 标签文件。"""

    if not path.is_file():
        raise FileNotFoundError(f"QMUL protocol file does not exist: {path}")
    try:
        from scipy.io import loadmat
    except ImportError as error:
        raise RuntimeError(
            "SciPy is required for QMUL protocol parsing; install the evaluation dependencies"
        ) from error
    return loadmat(path)


def _read_strings(payload: dict[str, Any], key: str) -> tuple[str, ...]:
    """从 MATLAB 列数组中读取字符串或数值身份编号。"""

    if key not in payload:
        raise ValueError(f"QMUL MAT file is missing field: {key}")
    values = payload[key].reshape(-1)
    return tuple(_scalar_to_string(value) for value in values)


def _scalar_to_string(value: Any) -> str:
    """把 MATLAB 标量、单元素数组或字节串统一转换为非空字符串。"""

    while hasattr(value, "shape") and getattr(value, "size", 0) == 1:
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    text = str(value)
    if not text:
        raise ValueError("QMUL protocol contains an empty identity or filename")
    return text


def _require_file(path: Path, relative_path: str) -> None:
    """确认官方协议引用的图片真实存在。"""

    if not path.is_file():
        raise FileNotFoundError(f"QMUL image referenced by protocol is missing: {relative_path}")
