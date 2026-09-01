from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CONFIG = """[recognition]
match_threshold = 0.45
min_margin = 0.05
top_k = 3

[recognition.quality_tiers.high]
match_threshold = 0.50
min_margin = 0.05

[recognition.quality_tiers.medium]
match_threshold = 0.60
min_margin = 0.08

[quality]
min_detection_score = 0.70
min_face_size_px = 112
min_blur_variance = 80.0
min_brightness = 35.0
max_brightness = 220.0
min_contrast = 12.0
high_quality_score = 0.75
medium_quality_score = 0.55
"""


@dataclass(frozen=True)
class QualityTierPolicy:
    """一个已通过质量检查的探针质量层级的开放集接收策略。"""

    match_threshold: float
    min_margin: float


@dataclass(frozen=True)
class Settings:
    """运行目录、质量规则和已标定识别参数。"""

    data_dir: Path
    config_path: Path
    database_path: Path
    faces_dir: Path
    models_dir: Path
    logs_dir: Path
    match_threshold: float
    min_margin: float
    top_k: int
    quality_tiers: dict[str, QualityTierPolicy]
    min_detection_score: float
    min_face_size_px: int
    min_blur_variance: float
    min_brightness: float
    max_brightness: float
    min_contrast: float
    high_quality_score: float
    medium_quality_score: float


def load_settings(data_dir: Path) -> Settings:
    """创建可迁移运行目录并读取当前 TOML 配置。

    参数：
        data_dir：保存数据库、图片、模型和日志的目录。
    返回：
        供应用各模块共享的不可变配置对象。
    前置条件：
        配置文件不存在时会生成当前版本的完整默认配置；已有配置必须包含当前字段。
    """

    resolved_data_dir = data_dir.expanduser().resolve()
    faces_dir = resolved_data_dir / "faces"
    models_dir = resolved_data_dir / "models"
    logs_dir = resolved_data_dir / "logs"
    for directory in (resolved_data_dir, faces_dir, models_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    config_path = resolved_data_dir / "config.toml"
    if not config_path.exists():
        config_path.write_text(DEFAULT_CONFIG, encoding="utf-8")

    with config_path.open("rb") as config_file:
        config = tomllib.load(config_file)
    recognition = config["recognition"]
    quality = config["quality"]
    tier_config = recognition["quality_tiers"]
    quality_tiers = {
        "high": _read_tier(tier_config["high"]),
        "medium": _read_tier(tier_config["medium"]),
    }
    return Settings(
        data_dir=resolved_data_dir,
        config_path=config_path,
        database_path=resolved_data_dir / "face_library.sqlite",
        faces_dir=faces_dir,
        models_dir=models_dir,
        logs_dir=logs_dir,
        match_threshold=float(recognition["match_threshold"]),
        min_margin=float(recognition["min_margin"]),
        top_k=int(recognition["top_k"]),
        quality_tiers=quality_tiers,
        min_detection_score=float(quality["min_detection_score"]),
        min_face_size_px=int(quality["min_face_size_px"]),
        min_blur_variance=float(quality["min_blur_variance"]),
        min_brightness=float(quality["min_brightness"]),
        max_brightness=float(quality["max_brightness"]),
        min_contrast=float(quality["min_contrast"]),
        high_quality_score=float(quality["high_quality_score"]),
        medium_quality_score=float(quality["medium_quality_score"]),
    )


def write_recognition_thresholds(
    settings: Settings,
    *,
    match_threshold: float,
    min_margin: float,
) -> None:
    """保存整体识别阈值，同时保留当前两个质量层级的策略。

    参数：
        settings：当前运行配置。
        match_threshold：最高候选得分阈值。
        min_margin：第一、第二候选的最小分差。
    前置条件：
        两个阈值都必须位于 `[0, 1]`。
    """

    _validate_thresholds(match_threshold, min_margin)
    _write_settings_file(
        settings,
        match_threshold=match_threshold,
        min_margin=min_margin,
        quality_tiers=settings.quality_tiers,
    )


def write_quality_tier_thresholds(
    settings: Settings,
    *,
    tier: str,
    match_threshold: float,
    min_margin: float,
) -> None:
    """保存一个探针质量层级的标定结果，不改变另一个层级的策略。

    参数：
        settings：当前运行配置。
        tier：要更新的质量层级，目前为 `high` 或 `medium`。
        match_threshold：该层级的最高候选得分阈值。
        min_margin：该层级的最小候选分差。
    前置条件：
        层级必须存在，且两个阈值都必须位于 `[0, 1]`。
    """

    if tier not in settings.quality_tiers:
        raise ValueError(f"unknown quality tier: {tier}")
    _validate_thresholds(match_threshold, min_margin)
    quality_tiers = {
        **settings.quality_tiers,
        tier: QualityTierPolicy(match_threshold=match_threshold, min_margin=min_margin),
    }
    _write_settings_file(
        settings,
        match_threshold=settings.match_threshold,
        min_margin=settings.min_margin,
        quality_tiers=quality_tiers,
    )


def _write_settings_file(
    settings: Settings,
    *,
    match_threshold: float,
    min_margin: float,
    quality_tiers: dict[str, QualityTierPolicy],
) -> None:
    """按当前完整字段重写配置文件，不保留旧配置字段。"""
    high_tier = quality_tiers["high"]
    medium_tier = quality_tiers["medium"]
    settings.config_path.write_text(
        "\n".join(
            (
                "[recognition]",
                f"match_threshold = {match_threshold:.6f}",
                f"min_margin = {min_margin:.6f}",
                f"top_k = {settings.top_k}",
                "",
                "[recognition.quality_tiers.high]",
                f"match_threshold = {high_tier.match_threshold:.6f}",
                f"min_margin = {high_tier.min_margin:.6f}",
                "",
                "[recognition.quality_tiers.medium]",
                f"match_threshold = {medium_tier.match_threshold:.6f}",
                f"min_margin = {medium_tier.min_margin:.6f}",
                "",
                "[quality]",
                f"min_detection_score = {settings.min_detection_score:.6f}",
                f"min_face_size_px = {settings.min_face_size_px}",
                f"min_blur_variance = {settings.min_blur_variance:.6f}",
                f"min_brightness = {settings.min_brightness:.6f}",
                f"max_brightness = {settings.max_brightness:.6f}",
                f"min_contrast = {settings.min_contrast:.6f}",
                f"high_quality_score = {settings.high_quality_score:.6f}",
                f"medium_quality_score = {settings.medium_quality_score:.6f}",
                "",
            )
        ),
        encoding="utf-8",
    )


def _validate_thresholds(match_threshold: float, min_margin: float) -> None:
    """检查识别阈值是否位于合法的闭区间 `[0, 1]`。"""
    if not 0.0 <= match_threshold <= 1.0:
        raise ValueError("match_threshold must be between 0 and 1")
    if not 0.0 <= min_margin <= 1.0:
        raise ValueError("min_margin must be between 0 and 1")


def _read_tier(raw_tier: dict[str, object]) -> QualityTierPolicy:
    """把 TOML 中的一个质量层级配置转换为类型化策略对象。"""
    return QualityTierPolicy(
        match_threshold=float(raw_tier["match_threshold"]),
        min_margin=float(raw_tier["min_margin"]),
    )
