from __future__ import annotations

import platform
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CameraDevice:
    """One camera index that OpenCV successfully opened."""

    index: int
    label: str


class CameraService:
    """Small cross-platform wrapper around OpenCV camera backends."""

    def __init__(self, *, cv2_module: Any | None = None, platform_name: str | None = None) -> None:
        if cv2_module is None:
            try:
                import cv2 as cv2_module
            except ImportError as error:
                raise RuntimeError("OpenCV is not installed; install the project dependencies first") from error
        self._cv2 = cv2_module
        self._platform_name = platform_name or platform.system()
        self._capture: Any | None = None

    def discover(self, *, max_index: int = 8) -> list[CameraDevice]:
        """Probe camera indices without assuming Linux /dev/video paths exist."""

        devices: list[CameraDevice] = []
        for index in range(max_index):
            capture = self._open_capture(index)
            if capture is None:
                continue
            capture.release()
            devices.append(CameraDevice(index=index, label=f"Camera {index}"))
        return devices

    def open(self, index: int) -> None:
        self.close()
        capture = self._open_capture(index)
        if capture is None:
            raise RuntimeError(f"camera {index} could not be opened")
        self._capture = capture

    def read_frame(self) -> np.ndarray:
        if self._capture is None:
            raise RuntimeError("no camera is open")
        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise RuntimeError("camera did not provide a frame")
        return frame

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def _open_capture(self, index: int) -> Any | None:
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
        if self._platform_name == "Windows":
            return (getattr(self._cv2, "CAP_DSHOW", None), None)
        if self._platform_name == "Linux":
            return (getattr(self._cv2, "CAP_V4L2", None), None)
        if self._platform_name == "Darwin":
            return (getattr(self._cv2, "CAP_AVFOUNDATION", None), None)
        return (None,)
