"""验证不同桌面环境下的 Qt 平台和输入法配置。"""

from __future__ import annotations

from camera_face_comparison.app import configure_qt_environment


def test_kde_wayland_startup_uses_native_wayland_input_method() -> None:
    """Linux Wayland 启动时应覆盖全局 Fcitx 插件模式，改用原生 Wayland 输入协议。"""

    environment = {
        "XDG_SESSION_TYPE": "wayland",
        "XDG_CURRENT_DESKTOP": "KDE",
        "QT_IM_MODULE": "fcitx",
    }

    configure_qt_environment(environment, platform_name="linux")

    assert environment["QT_QPA_PLATFORM"] == "wayland"
    assert environment["QT_IM_MODULE"] == "wayland"


def test_explicit_non_wayland_qt_platform_is_preserved() -> None:
    """显式选择 XCB 或离屏平台时不得强制改回 Wayland。"""

    environment = {
        "XDG_SESSION_TYPE": "wayland",
        "XDG_CURRENT_DESKTOP": "KDE",
        "QT_QPA_PLATFORM": "xcb",
        "QT_IM_MODULE": "fcitx",
    }

    configure_qt_environment(environment, platform_name="linux")

    assert environment["QT_QPA_PLATFORM"] == "xcb"
    assert environment["QT_IM_MODULE"] == "fcitx"


def test_other_platforms_are_not_modified() -> None:
    """Windows、macOS 和 Linux X11 会话必须继续使用各自的 Qt 默认配置。"""

    environment = {
        "XDG_SESSION_TYPE": "x11",
        "QT_IM_MODULE": "fcitx",
    }

    configure_qt_environment(environment, platform_name="linux")

    assert environment == {
        "XDG_SESSION_TYPE": "x11",
        "QT_IM_MODULE": "fcitx",
    }
