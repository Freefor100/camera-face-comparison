from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication, QMessageBox

from .camera import CameraService
from .config import load_settings
from .face_engine import FaceEngine
from .ui.main_window import MainWindow


def run_application(data_dir: Path) -> int:
    """启动图形界面。

    参数：
        data_dir：模型、数据库、样本图片和日志所在的数据目录。
    返回：
        Qt 应用退出码。
    前置条件：
        本地模型已经准备好，且当前环境安装了图形界面和推理依赖。
    """

    application = QApplication.instance() or QApplication(sys.argv)
    try:
        settings = load_settings(data_dir)
        face_engine = FaceEngine.from_local_model(settings)
        camera = CameraService()
        window = MainWindow(settings=settings, face_engine=face_engine, camera=camera)
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        QMessageBox.critical(application.activeWindow(), "启动失败", str(error))
        return 1
    window.show()
    return application.exec()
