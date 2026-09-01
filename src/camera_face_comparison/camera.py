from __future__ import annotations

import platform
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CameraDevice:
    """一个已经被 OpenCV 成功打开过的摄像头索引。"""

    index: int
    label: str


class CameraService:
    """封装 OpenCV 摄像头后端的跨平台服务。"""

    def __init__(self, *, cv2_module: Any | None = None, platform_name: str | None = None) -> None:
        """初始化摄像头服务。

        参数：
            cv2_module：可选的 OpenCV 替身，便于测试；省略时导入真实 OpenCV。
            platform_name：可选的平台名称；省略时读取当前操作系统。
        """
        if cv2_module is None:
            try:
                import cv2 as cv2_module
            except ImportError as error:
                raise RuntimeError("OpenCV is not installed; install the project dependencies first") from error
        self._cv2 = cv2_module
        self._platform_name = platform_name or platform.system()
        self._capture: Any | None = None

    def discover(self, *, max_index: int = 8) -> list[CameraDevice]:
        """扫描可打开的摄像头索引并返回设备列表。

        参数：
            max_index：扫描的索引上限，不包含该值。
        返回：
            能被当前平台后端打开的摄像头设备。
        前置条件：
            OpenCV 已安装；扫描只探测索引，不直接依赖 Linux 的 `/dev/video*` 路径。
        """

        devices: list[CameraDevice] = []
        for index in range(max_index):
            capture = self._open_capture(index)
            if capture is None:
                continue
            capture.release()
            devices.append(CameraDevice(index=index, label=f"Camera {index}"))
        return devices

    def open(self, index: int) -> None:
        """打开指定索引的摄像头，必要时先关闭当前设备。

        参数：
            index：操作系统/OpenCV 使用的摄像头索引。
        返回：
            无返回值；成功后服务持有可读的摄像头句柄。
        前置条件：
            索引对应的设备存在且当前进程有访问权限。
        """
        self.close()
        capture = self._open_capture(index)
        if capture is None:
            raise RuntimeError(f"camera {index} could not be opened")
        self._capture = capture

    def read_frame(self) -> np.ndarray:
        """从已打开的摄像头读取一帧 BGR 图像。

        返回：
            OpenCV 格式的三通道图像数组。
        前置条件：
            必须先成功调用 `open()`；摄像头此时仍能提供帧。
        """
        if self._capture is None:
            raise RuntimeError("no camera is open")
        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise RuntimeError("camera did not provide a frame")
        return frame

    def close(self) -> None:
        """释放当前摄像头句柄，使设备可以被再次打开。"""
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def _open_capture(self, index: int) -> Any | None:
        """按当前平台优先级尝试创建 OpenCV 捕获对象。"""
        for backend in self._backend_candidates():
            capture = (
                self._cv2.VideoCapture(index)
                if backend is None
                else self._cv2.VideoCapture(index, backend)
            )
            if capture.isOpened():
                return capture
            capture.release()
        return None

    def _backend_candidates(self) -> tuple[int | None, ...]:
        """返回当前平台应尝试的 OpenCV 后端顺序。"""
        if self._platform_name == "Windows":
            return (getattr(self._cv2, "CAP_DSHOW", None), None)
        if self._platform_name == "Linux":
            return (getattr(self._cv2, "CAP_V4L2", None), None)
        if self._platform_name == "Darwin":
            return (getattr(self._cv2, "CAP_AVFOUNDATION", None), None)
        return (None,)
