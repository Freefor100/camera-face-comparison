from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication, QMessageBox

from .camera import CameraService
from .config import load_settings
from .face_engine import FaceEngine
from .ui.main_window import MainWindow


def run_application(data_dir: Path) -> int:
    """Start the GUI after confirming that all local runtime assets are ready."""

    application = QApplication.instance() or QApplication(sys.argv)
    settings = load_settings(data_dir)
    try:
        face_engine = FaceEngine.from_local_model(settings)
        camera = CameraService()
    except RuntimeError as error:
        QMessageBox.critical(application.activeWindow(), "启动失败", str(error))
        return 1
    window = MainWindow(settings=settings, face_engine=face_engine, camera=camera)
    window.show()
    return application.exec()
