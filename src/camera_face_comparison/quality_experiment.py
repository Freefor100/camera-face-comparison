from __future__ import annotations

import hashlib
import json
import random
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path

from .lfw_dataset import LfwSplitProtocol


@dataclass(frozen=True)
class QualityKnownCase:
    """一个用于质量实验的 Known 身份、探针和参考图候选。"""

    person_id: str
    probe_path: str
    gallery_candidates: tuple[str, ...]


@dataclass(frozen=True)
class QualityUnknownCase:
    """一个不属于实验 Gallery 的 Unknown 来源身份和探针。"""

    source_identity: str
    probe_path: str


@dataclass(frozen=True)
class QualityExperimentProtocol:
    """只从 Calibration 构建的固定质量实验协议。"""

    known: tuple[QualityKnownCase, ...]
    unknown: tuple[QualityUnknownCase, ...]
    seed: int
    source_split_protocol_sha256: str


def build_quality_experiment_protocol(
    split_protocol: LfwSplitProtocol,
    decision_scores_path: Path,
    *,
    unknown_count: int = 300,
    seed: int = 2026,
) -> QualityExperimentProtocol:
    """从有效 Calibration Probe 构建身份互斥的质量实验协议。

    参数：
        split_protocol：Phase 2 固定的 Gallery 和 Probe 分区。
        decision_scores_path：Phase 2 无阈值分数数据库。
        unknown_count：最多选择多少个 Unknown 来源身份。
        seed：Unknown 身份抽样的固定随机种子。
    返回：
        每个来源身份最多一张 Probe 的确定性实验协议。
    前置条件：
        分数库必须存在且至少含一个有效 Calibration Known；Unknown 数量必须充足。
    """

    if unknown_count < 1:
        raise ValueError("unknown_count must be at least one")
    if not decision_scores_path.is_file():
        raise FileNotFoundError(f"decision score database does not exist: {decision_scores_path}")

    rows = _read_calibration_rows(decision_scores_path)
    known_paths: dict[str, list[str]] = {}
    unknown_paths: dict[str, list[str]] = {}
    for relative_path, source_identity, expected_person_id in rows:
        if expected_person_id is None:
            unknown_paths.setdefault(source_identity, []).append(relative_path)
        else:
            known_paths.setdefault(expected_person_id, []).append(relative_path)

    known = tuple(
        QualityKnownCase(
            person_id=person_id,
            probe_path=min(paths),
            gallery_candidates=split_protocol.enrollment[person_id],
        )
        for person_id, paths in sorted(known_paths.items())
        if split_protocol.enrollment.get(person_id)
    )
    if not known:
        raise ValueError("no valid Calibration Known probes were found")

    known_ids = {case.person_id for case in known}
    unknown_ids = sorted(identity for identity in unknown_paths if identity not in known_ids)
    if len(unknown_ids) < unknown_count:
        raise ValueError(
            f"not enough Calibration Unknown identities: need {unknown_count}, found {len(unknown_ids)}"
        )
    randomizer = random.Random(seed)
    randomizer.shuffle(unknown_ids)
    selected_unknown_ids = sorted(unknown_ids[:unknown_count])
    unknown = tuple(
        QualityUnknownCase(
            source_identity=source_identity,
            probe_path=min(unknown_paths[source_identity]),
        )
        for source_identity in selected_unknown_ids
    )
    return QualityExperimentProtocol(
        known=known,
        unknown=unknown,
        seed=seed,
        source_split_protocol_sha256=_split_protocol_sha256(split_protocol),
    )


def write_quality_experiment_protocol(
    protocol: QualityExperimentProtocol,
    output_path: Path,
) -> None:
    """把质量实验协议原子写入 JSON 文件。"""

    payload = {
        "protocol": "lfw-quality-experiment-v1",
        "seed": protocol.seed,
        "source_split_protocol_sha256": protocol.source_split_protocol_sha256,
        "known": [asdict(case) for case in protocol.known],
        "unknown": [asdict(case) for case in protocol.unknown],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)


def read_quality_experiment_protocol(path: Path) -> QualityExperimentProtocol:
    """严格读取当前版本的质量实验 JSON 协议。"""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["protocol"] != "lfw-quality-experiment-v1":
        raise ValueError("unsupported quality experiment protocol")
    known = tuple(
        QualityKnownCase(
            person_id=str(item["person_id"]),
            probe_path=str(item["probe_path"]),
            gallery_candidates=tuple(str(value) for value in item["gallery_candidates"]),
        )
        for item in payload["known"]
    )
    unknown = tuple(
        QualityUnknownCase(
            source_identity=str(item["source_identity"]),
            probe_path=str(item["probe_path"]),
        )
        for item in payload["unknown"]
    )
    return QualityExperimentProtocol(
        known=known,
        unknown=unknown,
        seed=int(payload["seed"]),
        source_split_protocol_sha256=str(payload["source_split_protocol_sha256"]),
    )


def _read_calibration_rows(path: Path) -> list[tuple[str, str, str | None]]:
    """从分数库读取每张有效 Calibration Probe 一次。"""

    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            """
            SELECT DISTINCT relative_path, source_identity, expected_person_id
            FROM decision_scores
            WHERE split = 'calibration' AND method = 'single' AND top_k = 0
            ORDER BY source_identity, relative_path
            """
        ).fetchall()
    finally:
        connection.close()
    return [
        (str(relative_path), str(source_identity), None if expected is None else str(expected))
        for relative_path, source_identity, expected in rows
    ]


def _split_protocol_sha256(protocol: LfwSplitProtocol) -> str:
    """计算与字典插入顺序无关的分区协议摘要。"""

    payload = {
        "enrollment": {
            person_id: list(paths)
            for person_id, paths in sorted(protocol.enrollment.items())
        },
        "probes": [asdict(probe) for probe in protocol.probes],
        "seed": protocol.seed,
        "calibration_fraction": protocol.calibration_fraction,
        "source_protocol_sha256": protocol.source_protocol_sha256,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
