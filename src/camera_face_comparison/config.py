from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CONFIG = """[recognition]
match_threshold = 0.45
min_margin = 0.05

[quality]
min_detection_score = 0.70
min_face_size_px = 112
min_blur_variance = 80.0
"""


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
    min_detection_score: float
    min_face_size_px: int
    min_blur_variance: float


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
    return Settings(
        data_dir=resolved_data_dir,
        config_path=config_path,
        database_path=resolved_data_dir / "face_library.sqlite",
        faces_dir=faces_dir,
        models_dir=models_dir,
        logs_dir=logs_dir,
        match_threshold=float(recognition["match_threshold"]),
        min_margin=float(recognition["min_margin"]),
        min_detection_score=float(quality["min_detection_score"]),
        min_face_size_px=int(quality["min_face_size_px"]),
        min_blur_variance=float(quality["min_blur_variance"]),
    )


def write_recognition_thresholds(
    settings: Settings,
    *,
    match_threshold: float,
    min_margin: float,
) -> None:
    """Persist a calibration result while retaining the active quality policy."""

    if not 0.0 <= match_threshold <= 1.0:
        raise ValueError("match_threshold must be between 0 and 1")
    if not 0.0 <= min_margin <= 1.0:
        raise ValueError("min_margin must be between 0 and 1")
    settings.config_path.write_text(
        "\n".join(
            (
                "[recognition]",
                f"match_threshold = {match_threshold:.6f}",
                f"min_margin = {min_margin:.6f}",
                "",
                "[quality]",
                f"min_detection_score = {settings.min_detection_score:.6f}",
                f"min_face_size_px = {settings.min_face_size_px}",
                f"min_blur_variance = {settings.min_blur_variance:.6f}",
                "",
            )
        ),
        encoding="utf-8",
    )
