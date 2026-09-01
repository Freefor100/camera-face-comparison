from __future__ import annotations

from camera_face_comparison.config import (
    load_settings,
    write_quality_tier_thresholds,
    write_recognition_thresholds,
)


def test_load_settings_creates_portable_data_layout(tmp_path) -> None:
    """Starting from an empty data directory must create every runtime location."""

    settings = load_settings(tmp_path)

    assert settings.data_dir == tmp_path
    assert settings.database_path == tmp_path / "face_library.sqlite"
    assert settings.faces_dir.is_dir()
    assert settings.models_dir.is_dir()
    assert settings.logs_dir.is_dir()
    assert settings.config_path.is_file()


def test_write_recognition_thresholds_persists_calibration_result(tmp_path) -> None:
    """Calibration must alter runtime behavior after the next application start."""

    settings = load_settings(tmp_path)
    write_recognition_thresholds(settings, match_threshold=0.61, min_margin=0.09)

    reloaded = load_settings(tmp_path)

    assert reloaded.match_threshold == 0.61
    assert reloaded.min_margin == 0.09
    assert "[enrollment]" not in settings.config_path.read_text(encoding="utf-8")


def test_settings_exposes_quality_tier_policy(tmp_path) -> None:
    """Open-set behavior must read quality-tier policy from portable config."""

    settings = load_settings(tmp_path)

    assert set(settings.quality_tiers) == {"high", "medium"}
    assert settings.quality_tiers["high"].match_threshold > 0
    assert (
        settings.quality_tiers["medium"].match_threshold
        >= settings.quality_tiers["high"].match_threshold
    )


def test_write_quality_tier_thresholds_keeps_the_other_probe_policy(tmp_path) -> None:
    """Medium-quality calibration must not silently overwrite the high-quality policy."""

    settings = load_settings(tmp_path)
    high_before = settings.quality_tiers["high"]

    write_quality_tier_thresholds(
        settings,
        tier="medium",
        match_threshold=0.63,
        min_margin=0.11,
    )

    reloaded = load_settings(tmp_path)
    assert reloaded.quality_tiers["high"] == high_before
    assert reloaded.quality_tiers["medium"].match_threshold == 0.63
    assert reloaded.quality_tiers["medium"].min_margin == 0.11
