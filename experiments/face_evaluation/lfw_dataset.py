from __future__ import annotations

import hashlib
import json
import random
import tarfile
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Literal, cast
from urllib.request import Request, urlopen

LFW_FUNNELED_URL = "https://ndownloader.figshare.com/files/5976015"
LFW_FUNNELED_SHA256 = "b47c8422c8cded889dc5a13418c4bc2abbda121092b3533a83306f90d900100a"


@dataclass(frozen=True)
class LfwProbe:
    """一张留出的 LFW 图片及其开放集评测标签。"""

    relative_path: str
    expected_person_id: str | None


@dataclass(frozen=True)
class LfwProtocol:
    """不复制图片、仅保存相对路径的确定性录入集与探针集划分。"""

    enrollment: dict[str, tuple[str, ...]]
    probes: tuple[LfwProbe, ...]


ProbeSplit = Literal["calibration", "evaluation"]


@dataclass(frozen=True)
class LfwSplitProbe:
    """一张带固定标定/评估分区及来源身份的 LFW 探针。"""

    relative_path: str
    expected_person_id: str | None
    source_identity: str
    split: ProbeSplit


@dataclass(frozen=True)
class LfwSplitProtocol:
    """在原开放集协议之上固定身份互斥的标定与评估分区。"""

    enrollment: dict[str, tuple[str, ...]]
    probes: tuple[LfwSplitProbe, ...]
    seed: int
    calibration_fraction: float
    source_protocol_sha256: str


def ensure_lfw_dataset(
    data_dir: Path,
    *,
    download: bool,
    downloader: Callable[[str, str], object] | None = None,
) -> Path:
    """返回本地 LFW 目录，仅在用户明确指定时下载数据集。

    参数：
        data_dir：数据集存放的运行数据目录。
        download：数据集不存在时是否允许联网下载。
        downloader：可注入的下载器，主要用于测试。
    返回：
        已验证的 LFW 图片目录。
    前置条件：
        不下载时目录或压缩包必须已存在；下载时网络源必须可用。
    """

    datasets_dir = data_dir / "datasets"
    target = datasets_dir / "lfw_funneled"
    if target.is_dir():
        return target
    archive = datasets_dir / "lfw-funneled.tgz"
    partial = archive.with_suffix(archive.suffix + ".part")
    if archive.is_file() and _sha256_file(archive) != LFW_FUNNELED_SHA256:
        if not download:
            raise RuntimeError("LFW archive checksum mismatch; rerun with --download to resume it")
        if partial.exists():
            raise RuntimeError("both an invalid LFW archive and a partial download exist")
        archive.replace(partial)
    if not archive.is_file():
        if not download:
            if partial.is_file():
                raise FileNotFoundError(
                    f"LFW download is incomplete at {partial}; rerun with --download to resume it"
                )
            raise FileNotFoundError(
                f"LFW is missing at {target}; rerun with --download on a networked machine"
            )
        datasets_dir.mkdir(parents=True, exist_ok=True)
        try:
            if downloader is None:
                _download_with_resume(LFW_FUNNELED_URL, partial)
                if _sha256_file(partial) != LFW_FUNNELED_SHA256:
                    raise RuntimeError("LFW archive checksum mismatch after download")
                partial.replace(archive)
            else:
                downloader(LFW_FUNNELED_URL, str(archive))
        except OSError as error:
            raise RuntimeError(
                "could not download LFW from the configured mirror; check the network and retry"
            ) from error
    if _sha256_file(archive) != LFW_FUNNELED_SHA256:
        raise RuntimeError("LFW archive checksum mismatch; do not extract it")
    _safe_extract(archive, datasets_dir)
    if not target.is_dir():
        raise RuntimeError(f"LFW archive did not create the expected directory: {target}")
    return target


def build_lfw_protocol(
    dataset_dir: Path,
    *,
    known_identity_count: int,
    unknown_identity_count: int,
    enrollment_per_identity: int,
    probes_per_identity: int,
) -> LfwProtocol:
    """从本地 LFW 身份生成确定性的开放集录入/探针划分。

    参数：
        dataset_dir：LFW 图片根目录。
        known_identity_count：进入标准库的身份数。
        unknown_identity_count：只作为未知探针的身份数。
        enrollment_per_identity：每个已知身份的录入图片数。
        probes_per_identity：每个身份的探针图片数。
    返回：
        已知身份和未知身份不相交的评测协议。
    前置条件：
        每个身份必须有足够图片，且计数参数必须为正数。
    """

    if min(
        known_identity_count,
        unknown_identity_count,
        enrollment_per_identity,
        probes_per_identity,
    ) < 1:
        raise ValueError("all LFW protocol counts must be at least one")
    required_known_images = enrollment_per_identity + probes_per_identity
    eligible_known = _identities_with_at_least(dataset_dir, required_known_images)
    eligible_unknown = _identities_with_at_least(dataset_dir, probes_per_identity)
    known_names = [identity.name for identity in eligible_known[:known_identity_count]]
    unknown_names = [
        identity.name for identity in eligible_unknown if identity.name not in set(known_names)
    ][:unknown_identity_count]
    if len(known_names) != known_identity_count or len(unknown_names) != unknown_identity_count:
        raise ValueError(
            "not enough LFW identities with the requested image counts for this open-set protocol"
        )

    enrollment: dict[str, tuple[str, ...]] = {}
    probes: list[LfwProbe] = []
    for name in known_names:
        images = _image_paths(dataset_dir / name)
        enrollment[name] = tuple(
            _relative_to_dataset(path, dataset_dir) for path in images[:enrollment_per_identity]
        )
        probes.extend(
            LfwProbe(_relative_to_dataset(path, dataset_dir), name)
            for path in images[enrollment_per_identity : enrollment_per_identity + probes_per_identity]
        )
    for name in unknown_names:
        probes.extend(
            LfwProbe(_relative_to_dataset(path, dataset_dir), None)
            for path in _image_paths(dataset_dir / name)[:probes_per_identity]
        )
    return LfwProtocol(enrollment=enrollment, probes=tuple(probes))


def build_full_lfw_protocol(
    dataset_dir: Path,
    *,
    known_fraction: float = 0.8,
    enrollment_per_identity: int = 5,
    seed: int = 2026,
) -> LfwProtocol:
    """构建覆盖 LFW 每张图片的开放集协议。

    参数：
        dataset_dir：LFW 图片根目录。
        known_fraction：分配到 Gallery 身份的比例，必须严格位于 0 和 1 之间。
        enrollment_per_identity：每个已知身份最多放入 Gallery 的图片数。
        seed：身份划分的固定随机种子。
    返回：
        覆盖每张 LFW 图片的开放集协议；已知身份的剩余图片是 Known Probe，
        未知身份的全部图片是 Unknown Probe。
    前置条件：
        数据集至少包含两个身份，且 `enrollment_per_identity` 为正数。
    """

    if not 0.0 < known_fraction < 1.0:
        raise ValueError("known_fraction must be between 0 and 1")
    if enrollment_per_identity < 1:
        raise ValueError("enrollment_per_identity must be at least one")
    identities = [
        path
        for path in sorted(dataset_dir.iterdir())
        if path.is_dir() and _image_paths(path)
    ] if dataset_dir.is_dir() else []
    if len(identities) < 2:
        raise ValueError("full LFW protocol requires at least two identities")

    shuffled = identities.copy()
    random.Random(seed).shuffle(shuffled)
    known_count = max(1, min(len(shuffled) - 1, round(len(shuffled) * known_fraction)))
    known_names = {identity.name for identity in shuffled[:known_count]}

    enrollment: dict[str, tuple[str, ...]] = {}
    probes: list[LfwProbe] = []
    for identity in identities:
        paths = _image_paths(identity)
        relative_paths = [_relative_to_dataset(path, dataset_dir) for path in paths]
        if identity.name in known_names:
            gallery_paths = relative_paths[:enrollment_per_identity]
            enrollment[identity.name] = tuple(gallery_paths)
            probes.extend(
                LfwProbe(path, identity.name) for path in relative_paths[enrollment_per_identity:]
            )
        else:
            probes.extend(LfwProbe(path, None) for path in relative_paths)
    return LfwProtocol(enrollment=enrollment, probes=tuple(probes))


def write_lfw_protocol(protocol: LfwProtocol, output_path: Path) -> None:
    """写出路径相对于 LFW 根目录的可迁移 JSON 评测协议。"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": "lfw-open-set-v1",
        "enrollment": {name: list(paths) for name, paths in protocol.enrollment.items()},
        "probes": [asdict(probe) for probe in protocol.probes],
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_lfw_protocol(path: Path) -> LfwProtocol:
    """读取当前版本的 LFW 开放集协议，不解析其他版本或旧字段。"""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol") != "lfw-open-set-v1":
        raise ValueError("unsupported LFW protocol file")
    enrollment = {
        str(person_id): tuple(str(relative_path) for relative_path in paths)
        for person_id, paths in payload["enrollment"].items()
    }
    probes = tuple(
        LfwProbe(
            relative_path=str(item["relative_path"]),
            expected_person_id=(
                None if item["expected_person_id"] is None else str(item["expected_person_id"])
            ),
        )
        for item in payload["probes"]
    )
    return LfwProtocol(enrollment=enrollment, probes=probes)


def lfw_protocol_sha256(protocol: LfwProtocol) -> str:
    """计算原始 LFW 协议的规范化 SHA-256，绑定后续实验数据。"""

    payload = {
        "protocol": "lfw-open-set-v1",
        "enrollment": {name: list(paths) for name, paths in sorted(protocol.enrollment.items())},
        "probes": [asdict(probe) for probe in protocol.probes],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def split_lfw_protocol(
    protocol: LfwProtocol,
    *,
    seed: int = 2026,
    calibration_fraction: float = 0.5,
) -> LfwSplitProtocol:
    """按身份固定划分 Calibration 与 Evaluation 探针。

    参数：
        protocol：质量过滤前生成的原始开放集协议。
        seed：控制身份洗牌的固定随机种子。
        calibration_fraction：每类探针身份分配给 Calibration 的比例。
    返回：
        Gallery 不变、Known 与 Unknown 来源身份均不跨分区的固定协议。
    前置条件：
        比例必须严格位于 0 和 1 之间；协议中的探针路径必须含身份目录。
    """

    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be between 0 and 1")

    known_identities = sorted(
        {probe.expected_person_id for probe in protocol.probes if probe.expected_person_id is not None}
    )
    unknown_identities = sorted(
        {_probe_source_identity(probe) for probe in protocol.probes if probe.expected_person_id is None}
    )
    assignments: dict[tuple[str, str], ProbeSplit] = {}
    for role, identities in (("known", known_identities), ("unknown", unknown_identities)):
        shuffled = list(identities)
        # 为 Known/Unknown 使用独立且稳定的随机流，避免一组数量变化扰动另一组。
        role_seed = int.from_bytes(hashlib.sha256(f"{seed}:{role}".encode()).digest()[:8], "big")
        random.Random(role_seed).shuffle(shuffled)
        calibration_count = _calibration_identity_count(
            len(shuffled), calibration_fraction=calibration_fraction
        )
        calibration_identities = set(shuffled[:calibration_count])
        assignments.update(
            {
                (role, identity): (
                    "calibration" if identity in calibration_identities else "evaluation"
                )
                for identity in shuffled
            }
        )

    split_probes = tuple(
        LfwSplitProbe(
            relative_path=probe.relative_path,
            expected_person_id=probe.expected_person_id,
            source_identity=_probe_source_identity(probe),
            split=assignments[
                (
                    "known" if probe.expected_person_id is not None else "unknown",
                    _probe_source_identity(probe),
                )
            ],
        )
        for probe in protocol.probes
    )
    return LfwSplitProtocol(
        enrollment=dict(protocol.enrollment),
        probes=split_probes,
        seed=seed,
        calibration_fraction=calibration_fraction,
        source_protocol_sha256=lfw_protocol_sha256(protocol),
    )


def write_lfw_split_protocol(protocol: LfwSplitProtocol, output_path: Path) -> None:
    """以原子替换方式保存固定 LFW 标定/评估协议。"""

    payload = {
        "protocol": "lfw-decision-split-v1",
        "seed": protocol.seed,
        "calibration_fraction": protocol.calibration_fraction,
        "source_protocol_sha256": protocol.source_protocol_sha256,
        "enrollment": {name: list(paths) for name, paths in protocol.enrollment.items()},
        "probes": [asdict(probe) for probe in protocol.probes],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_path)


def read_lfw_split_protocol(path: Path) -> LfwSplitProtocol:
    """读取当前版本的 LFW 标定/评估固定分区协议。"""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol") != "lfw-decision-split-v1":
        raise ValueError("unsupported LFW decision split protocol file")
    probes: list[LfwSplitProbe] = []
    for item in payload["probes"]:
        split = str(item["split"])
        if split not in {"calibration", "evaluation"}:
            raise ValueError(f"unsupported LFW probe split: {split}")
        probes.append(
            LfwSplitProbe(
                relative_path=str(item["relative_path"]),
                expected_person_id=(
                    None
                    if item["expected_person_id"] is None
                    else str(item["expected_person_id"])
                ),
                source_identity=str(item["source_identity"]),
                split=cast(ProbeSplit, split),
            )
        )
    return LfwSplitProtocol(
        enrollment={
            str(person_id): tuple(str(relative_path) for relative_path in paths)
            for person_id, paths in payload["enrollment"].items()
        },
        probes=tuple(probes),
        seed=int(payload["seed"]),
        calibration_fraction=float(payload["calibration_fraction"]),
        source_protocol_sha256=str(payload["source_protocol_sha256"]),
    )


def _probe_source_identity(probe: LfwProbe) -> str:
    """返回探针真实来源身份；Unknown 从相对路径首级目录恢复。"""

    if probe.expected_person_id is not None:
        return probe.expected_person_id
    parts = PurePosixPath(probe.relative_path).parts
    if len(parts) < 2 or not parts[0]:
        raise ValueError(f"probe path does not contain an identity directory: {probe.relative_path}")
    return parts[0]


def _calibration_identity_count(total: int, *, calibration_fraction: float) -> int:
    """计算身份分区数量；有至少两个身份时保证两侧均非空。"""

    if total == 0:
        return 0
    if total == 1:
        return 1
    return max(1, min(total - 1, round(total * calibration_fraction)))


def _identities_with_at_least(dataset_dir: Path, minimum_images: int) -> list[Path]:
    """筛选出图片数量达到要求的 LFW 身份，并按名称排序。"""
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"LFW directory does not exist: {dataset_dir}")
    return [
        identity
        for identity in sorted(path for path in dataset_dir.iterdir() if path.is_dir())
        if len(_image_paths(identity)) >= minimum_images
    ]


def _image_paths(identity_dir: Path) -> list[Path]:
    """返回一个身份目录下按文件名排序的图片路径。"""
    return sorted(
        path
        for path in identity_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )


def _relative_to_dataset(path: Path, dataset_dir: Path) -> str:
    """把数据集内的绝对路径转换为可写入协议的相对路径。"""
    return path.relative_to(dataset_dir).as_posix()


def _safe_extract(archive_path: Path, destination: Path) -> None:
    """校验压缩包成员路径后再安全解压，防止路径穿越。"""
    destination_root = destination.resolve()
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            member_path = (destination_root / member.name).resolve()
            try:
                member_path.relative_to(destination_root)
            except ValueError as error:
                raise RuntimeError(f"unsafe archive member: {member.name}") from error
        archive.extractall(destination_root, filter="data")


def _download_with_resume(url: str, partial_path: Path) -> None:
    """使用 HTTP Range 从临时文件大小处继续下载数据集压缩包。"""
    offset = partial_path.stat().st_size if partial_path.exists() else 0
    request = Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
    with urlopen(request, timeout=60) as response:
        append = offset > 0 and response.status == 206
        with partial_path.open("ab" if append else "wb") as output_file:
            while chunk := response.read(1024 * 1024):
                output_file.write(chunk)


def _sha256_file(path: Path) -> str:
    """分块计算数据集压缩包或文件的 SHA-256 哈希。"""
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
