from __future__ import annotations

import numpy as np

from camera_face_comparison.camera import CameraService


class FakeCapture:
    """用于验证摄像头服务生命周期的最小 OpenCV 捕获替身。"""

    def __init__(self, opened: bool, frame: np.ndarray | None = None) -> None:
        """初始化可配置的打开状态和待返回帧。"""
        self.opened = opened
        self.frame = frame
        self.released = False

    def isOpened(self) -> bool:
        """返回替身是否模拟为已打开。"""
        return self.opened

    def read(self) -> tuple[bool, np.ndarray | None]:
        """返回预置帧及其是否可用的标志。"""
        return self.frame is not None, self.frame

    def release(self) -> None:
        """记录服务是否释放了该捕获对象。"""
        self.released = True


class FakeCv2:
    """记录后端调用并为测试提供可控摄像头设备的 OpenCV 替身。"""

    CAP_AVFOUNDATION = 1200
    CAP_DSHOW = 700
    CAP_V4L2 = 200

    def __init__(self) -> None:
        """初始化调用记录和捕获对象记录。"""
        self.calls: list[tuple[int, int | None]] = []
        self.captures: list[FakeCapture] = []

    def VideoCapture(self, index: int, backend: int | None = None) -> FakeCapture:
        """根据索引返回测试捕获对象并记录所用后端。"""
        self.calls.append((index, backend))
        frame = np.zeros((10, 10, 3), dtype=np.uint8) if index == 1 else None
        capture = FakeCapture(opened=index == 1, frame=frame)
        self.captures.append(capture)
        return capture


def test_camera_service_scans_openable_indices_and_reads_selected_camera() -> None:
    """平台相关后端选择必须被封装在小型摄像头服务接口之后。"""

    fake_cv2 = FakeCv2()
    service = CameraService(cv2_module=fake_cv2, platform_name="Linux")

    assert [device.index for device in service.discover(max_index=3)] == [1]
    service.open(1)
    frame = service.read_frame()
    service.close()

    assert frame.shape == (10, 10, 3)
    assert any(backend == fake_cv2.CAP_V4L2 for _, backend in fake_cv2.calls)
    assert any(capture.released for capture in fake_cv2.captures)


def test_camera_service_prefers_avfoundation_on_macos() -> None:
    """macOS 应优先尝试 AVFoundation，再回退到通用后端。"""
    fake_cv2 = FakeCv2()
    service = CameraService(cv2_module=fake_cv2, platform_name="Darwin")

    assert service._backend_candidates() == (fake_cv2.CAP_AVFOUNDATION, None)
