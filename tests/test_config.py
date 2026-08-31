from __future__ import annotations

from camera_face_comparison.config import load_settings, write_recognition_thresholds


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
