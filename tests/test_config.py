from __future__ import annotations

import pytest

from camera_face_comparison.config import load_settings
from camera_face_comparison.open_set_policy import ScoreThresholdPolicy


def test_load_settings_creates_portable_data_layout_and_frozen_policy(tmp_path) -> None:
    """空数据目录必须生成完整布局和实验冻结的部署策略。"""

    settings = load_settings(tmp_path)

    assert settings.data_dir == tmp_path
    assert settings.database_path == tmp_path / "face_library.sqlite"
    assert settings.faces_dir.is_dir()
    assert settings.models_dir.is_dir()
    assert settings.logs_dir.is_dir()
    assert settings.config_path.is_file()
    assert settings.aggregation_method == "mean_prototype"
    assert isinstance(settings.recognition_policy, ScoreThresholdPolicy)
    assert settings.recognition_policy.minimum_score == pytest.approx(0.5557855367660522)


def test_load_settings_rejects_incomplete_configuration(tmp_path) -> None:
    """缺失当前必需字段的配置必须明确失败。"""

    config_path = tmp_path / "config.toml"
    tmp_path.mkdir(exist_ok=True)
    config_path.write_text(
        '[recognition]\naggregation_method = "mean_prototype"\n',
        encoding="utf-8",
    )

    with pytest.raises((KeyError, ValueError)):
        load_settings(tmp_path)
