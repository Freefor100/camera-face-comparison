from __future__ import annotations

import hashlib
import json
import tarfile
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
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
    legacy_target = datasets_dir / "lfw-deepfunneled"
    if target.is_dir():
        return target
    if legacy_target.is_dir():
        return legacy_target
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


def write_lfw_protocol(protocol: LfwProtocol, output_path: Path) -> None:
    """写出路径相对于 LFW 根目录的可迁移 JSON 评测协议。"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": "lfw-open-set-v1",
        "enrollment": {name: list(paths) for name, paths in protocol.enrollment.items()},
        "probes": [asdict(probe) for probe in protocol.probes],
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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
