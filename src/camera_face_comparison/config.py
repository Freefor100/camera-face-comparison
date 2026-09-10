from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .open_set_policy import ScoreThresholdPolicy

DEFAULT_CONFIG = """[recognition]
aggregation_method = "mean_prototype"
rule = "score_threshold"
minimum_score = 0.5557855367660522

[quality_warnings]
low_detection_score = 0.70
small_face_size_px = 112
low_blur_variance = 80.0
low_brightness = 35.0
high_brightness = 220.0
low_contrast = 12.0
"""


@dataclass(frozen=True)
class QualityWarningThresholds:
    """把原始图像指标转换为非阻断操作提示的参考界限。"""

    low_detection_score: float
    small_face_size_px: int
    low_blur_variance: float
    low_brightness: float
    high_brightness: float
    low_contrast: float


@dataclass(frozen=True)
class Settings:
    """运行目录、冻结识别策略和非阻断质量提示配置。"""

    data_dir: Path
    config_path: Path
    database_path: Path
    faces_dir: Path
    models_dir: Path
    logs_dir: Path
    aggregation_method: Literal["mean_prototype"]
    recognition_policy: ScoreThresholdPolicy
    quality_warnings: QualityWarningThresholds


def load_settings(data_dir: Path) -> Settings:
    """创建运行目录并严格读取当前配置格式。

    参数：
        data_dir：保存数据库、图片、模型和日志的目录。
    返回：
        当前运行目录、Mean Prototype 聚合、开放集策略和提示界限。
    前置条件：
        已有配置必须只包含当前字段；本项目不解析旧质量层级或双阈值配置。
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

    _require_exact_keys(config, {"recognition", "quality_warnings"}, "root")
    recognition = _required_table(config, "recognition")
    warning_values = _required_table(config, "quality_warnings")
    aggregation_method = recognition.get("aggregation_method")
    if aggregation_method != "mean_prototype":
        raise ValueError("recognition.aggregation_method must be mean_prototype")
    policy = _read_recognition_policy(recognition)
    _require_exact_keys(
        warning_values,
        {
            "low_detection_score",
            "small_face_size_px",
            "low_blur_variance",
            "low_brightness",
            "high_brightness",
            "low_contrast",
        },
        "quality_warnings",
    )
    warnings = QualityWarningThresholds(
        low_detection_score=float(warning_values["low_detection_score"]),
        small_face_size_px=int(warning_values["small_face_size_px"]),
        low_blur_variance=float(warning_values["low_blur_variance"]),
        low_brightness=float(warning_values["low_brightness"]),
        high_brightness=float(warning_values["high_brightness"]),
        low_contrast=float(warning_values["low_contrast"]),
    )
    _validate_warning_thresholds(warnings)
    return Settings(
        data_dir=resolved_data_dir,
        config_path=config_path,
        database_path=resolved_data_dir / "face_library.sqlite",
        faces_dir=faces_dir,
        models_dir=models_dir,
        logs_dir=logs_dir,
        aggregation_method="mean_prototype",
        recognition_policy=policy,
        quality_warnings=warnings,
    )


def _read_recognition_policy(values: dict[str, object]) -> ScoreThresholdPolicy:
    """读取已由联合标定冻结的最高分阈值规则。"""

    rule = values.get("rule")
    if rule != "score_threshold":
        raise ValueError("recognition.rule must be score_threshold")
    _require_exact_keys(
        values,
        {"aggregation_method", "rule", "minimum_score"},
        "recognition",
    )
    return ScoreThresholdPolicy(minimum_score=float(values["minimum_score"]))


def _required_table(config: dict[str, object], name: str) -> dict[str, object]:
    """读取必需 TOML 表，并在类型错误时给出明确位置。"""

    value = config[name]
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a TOML table")
    return value


def _require_exact_keys(
    values: dict[str, object],
    expected: set[str],
    location: str,
) -> None:
    """要求配置节点字段与当前格式完全一致。"""

    actual = set(values)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(
            f"invalid {location} fields; missing={missing}, unexpected={unexpected}"
        )


def _validate_warning_thresholds(values: QualityWarningThresholds) -> None:
    """检查非阻断提示界限是否处于可解释范围。"""

    if not 0.0 <= values.low_detection_score <= 1.0:
        raise ValueError("quality_warnings.low_detection_score must be between 0 and 1")
    if values.small_face_size_px < 1:
        raise ValueError("quality_warnings.small_face_size_px must be positive")
    if min(values.low_blur_variance, values.low_brightness, values.low_contrast) < 0.0:
        raise ValueError("quality warning lower bounds must not be negative")
    if values.high_brightness <= values.low_brightness:
        raise ValueError("high_brightness must exceed low_brightness")
