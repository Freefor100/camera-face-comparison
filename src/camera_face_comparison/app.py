"""组装应用依赖、配置 Qt 运行环境并管理桌面应用生命周期。"""

from __future__ import annotations

import os
import sys
from collections.abc import MutableMapping
from pathlib import Path

from PySide6.QtWidgets import QApplication, QMessageBox

from .camera import CameraService
from .config import load_settings
from .face_engine import FaceEngine
from .ui.main_window import MainWindow


def configure_qt_environment(
    environment: MutableMapping[str, str] | None = None,
    *,
    platform_name: str | None = None,
) -> None:
    """为 KDE Wayland 会话选择原生 Qt 窗口和输入法协议。

    参数：
        environment：要修改的环境变量映射；默认使用当前进程环境。
        platform_name：操作系统平台名称；默认使用当前 Python 平台。
    前置条件：
        只有 Linux KDE Wayland 且没有显式选择其他 Qt 平台时才会修改变量。
        明确指定的 XCB、离屏测试或其他平台配置始终保留。
    """

    values = os.environ if environment is None else environment
    current_platform = sys.platform if platform_name is None else platform_name
    if current_platform != "linux":
        return
    if values.get("XDG_SESSION_TYPE", "").casefold() != "wayland":
        return
    desktop = values.get("XDG_CURRENT_DESKTOP", "").casefold()
    is_kde = "kde" in desktop.split(":") or values.get("KDE_FULL_SESSION") == "true"
    if not is_kde:
        return

    qt_platform = values.get("QT_QPA_PLATFORM")
    if qt_platform is not None and qt_platform not in {"wayland", "wayland-egl"}:
        return
    values["QT_QPA_PLATFORM"] = qt_platform or "wayland"
    # pip 安装的 PySide6 不附带 Fcitx5 Qt 插件。KWin 可以通过 Wayland text-input
    # 协议把预编辑和提交文本转交给 Fcitx5，因此这里覆盖桌面全局的 fcitx 插件模式。
    values["QT_IM_MODULE"] = "wayland"


def run_application(data_dir: Path) -> int:
    """启动图形界面。

    参数：
        data_dir：模型、数据库、样本图片和日志所在的数据目录。
    返回：
        Qt 应用退出码。
    前置条件：
        本地模型已经准备好，且当前环境安装了图形界面和推理依赖。
    """

    configure_qt_environment()
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
