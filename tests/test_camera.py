from __future__ import annotations

import numpy as np

from camera_face_comparison.camera import CameraService


class FakeCapture:
    def __init__(self, opened: bool, frame: np.ndarray | None = None) -> None:
        self.opened = opened
        self.frame = frame
        self.released = False

    def isOpened(self) -> bool:
        return self.opened

    def read(self) -> tuple[bool, np.ndarray | None]:
        return self.frame is not None, self.frame

    def release(self) -> None:
        self.released = True


class FakeCv2:
    CAP_AVFOUNDATION = 1200
    CAP_DSHOW = 700
    CAP_V4L2 = 200

    def __init__(self) -> None:
        self.calls: list[tuple[int, int | None]] = []
        self.captures: list[FakeCapture] = []

    def VideoCapture(self, index: int, backend: int | None = None) -> FakeCapture:
        self.calls.append((index, backend))
        frame = np.zeros((10, 10, 3), dtype=np.uint8) if index == 1 else None
        capture = FakeCapture(opened=index == 1, frame=frame)
        self.captures.append(capture)
        return capture


def test_camera_service_scans_openable_indices_and_reads_selected_camera() -> None:
    """Platform-specific backends must stay behind a small camera-service API."""

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
    fake_cv2 = FakeCv2()
    service = CameraService(cv2_module=fake_cv2, platform_name="Darwin")

    assert service._backend_candidates() == (fake_cv2.CAP_AVFOUNDATION, None)
