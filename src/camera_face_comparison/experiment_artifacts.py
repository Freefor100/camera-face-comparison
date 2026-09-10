from __future__ import annotations

import hashlib
import json
from pathlib import Path


def file_sha256(path: Path) -> str:
    """分块计算文件 SHA-256，供实验缓存与产物清单校验内容。"""

    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def embedding_extraction_id(base_model: str = "buffalo_l") -> str:
    """生成只描述检测、对齐和特征提取实现的稳定标识。

    判定阈值、人员聚合和质量提示不会改变模型输出，因此不进入该标识。
    """

    payload = {
        "base_model": base_model,
        "detector_input_size": [640, 640],
        "alignment": "insightface-default-five-point",
        "embedding_normalization": "l2",
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"{base_model}:{hashlib.sha256(encoded).hexdigest()[:16]}"


def write_json_atomic(path: Path, payload: object) -> None:
    """先写临时文件再原子替换 JSON，避免中断后留下半截报告。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary_path.replace(path)
