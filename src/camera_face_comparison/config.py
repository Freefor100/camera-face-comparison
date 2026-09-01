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
    """Open-set acceptance policy for one validated probe-quality tier."""

    match_threshold: float
    min_margin: float


@dataclass(frozen=True)
class Settings:
    """Filesystem locations and calibrated recognition settings."""

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
    """Create the portable runtime layout and load its TOML configuration."""

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
    tier_config = recognition.get("quality_tiers", {})
    quality_tiers = {
        "high": _read_tier(
            tier_config.get("high", {}),
            fallback_threshold=float(recognition["match_threshold"]),
            fallback_margin=float(recognition["min_margin"]),
        ),
        "medium": _read_tier(
            tier_config.get("medium", {}),
            fallback_threshold=float(recognition["match_threshold"]),
            fallback_margin=float(recognition["min_margin"]),
        ),
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
        top_k=int(recognition.get("top_k", 3)),
        quality_tiers=quality_tiers,
        min_detection_score=float(quality["min_detection_score"]),
        min_face_size_px=int(quality["min_face_size_px"]),
        min_blur_variance=float(quality["min_blur_variance"]),
        min_brightness=float(quality.get("min_brightness", 35.0)),
        max_brightness=float(quality.get("max_brightness", 220.0)),
        min_contrast=float(quality.get("min_contrast", 12.0)),
        high_quality_score=float(quality.get("high_quality_score", 0.75)),
        medium_quality_score=float(quality.get("medium_quality_score", 0.55)),
    )


def write_recognition_thresholds(
    settings: Settings,
    *,
    match_threshold: float,
    min_margin: float,
) -> None:
    """Persist a calibration result while retaining the active quality policy."""

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
    """Persist calibration for one probe-quality tier without weakening the other tier."""

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
    if not 0.0 <= match_threshold <= 1.0:
        raise ValueError("match_threshold must be between 0 and 1")
    if not 0.0 <= min_margin <= 1.0:
        raise ValueError("min_margin must be between 0 and 1")


def _read_tier(
    raw_tier: dict[str, object],
    *,
    fallback_threshold: float,
    fallback_margin: float,
) -> QualityTierPolicy:
    return QualityTierPolicy(
        match_threshold=float(raw_tier.get("match_threshold", fallback_threshold)),
        min_margin=float(raw_tier.get("min_margin", fallback_margin)),
    )
