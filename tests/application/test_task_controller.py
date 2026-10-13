"""验证任务异常、数据库恢复与界面提示的实际业务边界。"""

import os
import sqlite3
import time
from threading import Event

import cv2
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication, QMessageBox

from camera_face_comparison.camera import CameraDevice
from camera_face_comparison.config import load_settings
from camera_face_comparison.face_engine import DetectedFace, FaceObservation
from camera_face_comparison.repository import FaceRepository, SampleInput
from camera_face_comparison.runtime import ExecutionBackend
from camera_face_comparison.ui.main_window import MainWindow
from tests.application.helpers import embedding


class Camera:
    """提供正常读取且可退出的摄像头，不访问真实设备。"""

    def discover(self):
        """返回可采集的固定设备。"""
        return [CameraDevice(0, "测试摄像头")]

    def open(self, index):
        """模拟打开设备。"""

    def read_frame(self):
        """返回独立的正常视频帧。"""
        return np.full((180, 180, 3), 120, dtype=np.uint8)

    def close(self):
        """模拟释放设备。"""


class Engine:
    """仅替换昂贵模型，保留识别、数据库与界面处理。"""

    backend = ExecutionBackend("cpu", ("CPUExecutionProvider",), -1)

    def detect_single_face(self, frame):
        """返回一张可用单脸。"""
        return DetectedFace((0, 0, 160, 160), 0.95, 100, np.zeros((5, 2)))

    def extract_detected_face(self, frame, detected):
        """返回确定的生产维度单位特征。"""
        return FaceObservation(detected.bbox, 0.95, embedding([1, 0]), 100, detected.landmarks)

    def extract_single_face(self, frame):
        """执行与生产接口一致的单帧提取。"""
        return self.extract_detected_face(frame, self.detect_single_face(frame))


def wait_for(application, condition):
    """处理实际 Qt 信号直到条件满足，超时即报告真实失败。"""
    deadline = time.monotonic() + 4
    while not condition() and time.monotonic() < deadline:
        application.processEvents()
        time.sleep(0.001)
    assert condition()


@pytest.fixture
def app():
    """复用离屏 Qt 应用。"""
    return QApplication.instance() or QApplication([])


def window_at(root, engine=None):
    """创建完整窗口而非控件替身。"""
    return MainWindow(settings=load_settings(root), face_engine=engine or Engine(), camera=Camera())


def test_corrupt_database_keeps_window_and_recovers_after_external_restore(tmp_path, app):
    """坏数据库不阻止窗口构建，恢复后必须能通过复查重新启用识别。"""
    settings = load_settings(tmp_path)
    settings.database_path.write_bytes(b"not a database")
    window = window_at(tmp_path)
    wait_for(app, lambda: window.controller.idle)
    assert not window.controller.library_valid
    assert not window.import_compare_button.isEnabled()
    settings.database_path.unlink()
    FaceRepository(settings.database_path).close()
    window.recheck_library()
    wait_for(app, lambda: window.controller.idle)
    assert window.controller.library_valid
    window.close()
    wait_for(app, lambda: window.controller.can_close)


def test_camera_error_stops_collection_and_does_not_leave_healthy_status(
    tmp_path, app, monkeypatch
):
    """识别提前失败必须停止采集、撤销健康状态并退出当前任务。"""
    window = window_at(tmp_path)
    wait_for(app, lambda: window.controller.idle)
    window.start_camera()
    wait_for(app, lambda: window.controller.current_frame is not None)
    monkeypatch.setattr(
        "camera_face_comparison.ui.workers.FaceRepository",
        lambda *args: (_ for _ in ()).throw(sqlite3.DatabaseError("malformed")),
    )
    window.compare_current_frame()
    wait_for(app, lambda: window.controller.idle)
    assert window.controller.capture_timer is None
    assert not window.controller.library_valid
    assert "失败" in window.result_label.text()
    window.close()
    wait_for(app, lambda: window.controller.can_close)


def test_delete_clears_result_and_reports_only_in_library_page(tmp_path, app, monkeypatch):
    """标准库变更后旧结果失效，删除消息不能写进识别卡片。"""
    import hashlib

    settings = load_settings(tmp_path)
    repository = FaceRepository(settings.database_path)
    photo = settings.faces_dir / "a" / "sample.jpg"
    photo.parent.mkdir()
    cv2.imwrite(str(photo), np.full((180, 180, 3), 120, dtype=np.uint8))
    repository.create_person_with_samples(
        person_id="a",
        display_name="甲",
        samples=[
            SampleInput(
                "faces/a/sample.jpg",
                embedding([1, 0]),
                {},
                image_sha256=hashlib.sha256(photo.read_bytes()).hexdigest(),
            )
        ],
    )
    repository.close()
    window = window_at(tmp_path)
    wait_for(app, lambda: window.controller.idle)
    window.controller.recognize_file(photo)
    wait_for(app, lambda: window.controller.idle)
    assert window.result_label.property("state") == "matched"
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.StandardButton.Yes)
    window.delete_selected_person()
    wait_for(app, lambda: window.controller.idle)
    assert window.result_label.property("state") == "idle"
    assert "删除" in window._library_page.status_label.text()
    assert "删除" not in window.status_label.text()
    assert not photo.exists()
    window.close()
    wait_for(app, lambda: window.controller.can_close)


def test_log_failure_keeps_completed_recognition_result(tmp_path, app, monkeypatch):
    """日志故障只能附带警告，不能把已经计算的未知判定改成系统失败。"""
    window = window_at(tmp_path)
    wait_for(app, lambda: window.controller.idle)
    photo = tmp_path / "probe.jpg"
    cv2.imwrite(str(photo), np.full((180, 180, 3), 120, dtype=np.uint8))

    def fail_log(*args, **kwargs):
        """在真正写日志的边界注入故障。"""
        raise sqlite3.DatabaseError("malformed")

    monkeypatch.setattr(FaceRepository, "record_recognition", fail_log)
    window.controller.recognize_file(photo)
    wait_for(app, lambda: window.controller.idle)
    assert window.result_label.property("state") == "unknown"
    assert "日志" in window._recognition_page.log_warning_label.text()
    assert not window.controller.library_valid
    window.close()
    wait_for(app, lambda: window.controller.can_close)


def test_close_waits_for_inference_without_blocking_the_ui(tmp_path, app):
    """模型尚未返回时关闭必须延迟，返回后才允许释放窗口。"""
    entered, release = Event(), Event()

    class BlockingEngine(Engine):
        """模拟暂时阻塞但最终返回的模型。"""

        def detect_single_face(self, frame):
            """等待测试释放，不访问真实模型。"""
            entered.set()
            release.wait(3)
            return super().detect_single_face(frame)

    window = window_at(tmp_path, BlockingEngine())
    wait_for(app, lambda: window.controller.idle)
    photo = tmp_path / "probe.jpg"
    cv2.imwrite(str(photo), np.ones((180, 180, 3), dtype=np.uint8))
    window.controller.recognize_file(photo)
    wait_for(app, entered.is_set)
    window.close()
    assert not window.controller.can_close
    release.set()
    wait_for(app, lambda: window.controller.can_close)


def test_late_task_error_and_late_camera_frame_cannot_change_current_view(tmp_path, app):
    """已结束任务的错误和停止后的视频帧不能覆盖当前显示。"""
    from camera_face_comparison.errors import TaskFailure

    window = window_at(tmp_path)
    wait_for(app, lambda: window.controller.idle)
    window.controller._on_error(-1, TaskFailure("storage_unavailable", "旧任务", "迟到"))
    assert window.controller.library_valid
    window.start_camera()
    wait_for(app, lambda: window.controller.current_frame is not None)
    old_worker = window.controller.camera_worker
    window.stop_camera()
    wait_for(app, lambda: window.controller.camera_worker is None)
    window.controller._camera_frame(old_worker, np.zeros((20, 20, 3), dtype=np.uint8))
    assert window.controller.current_frame is None
    assert window.preview_label.pixmap().isNull()
    assert "已打开" not in window._recognition_page.device_status_label.text()
    window.close()


def test_rollback_uncertainty_disables_business_until_recheck(tmp_path, app, monkeypatch):
    """业务冲突后的回滚失败必须撤销健康状态，而不是仅提示重复图片。"""
    from camera_face_comparison.image_input import ImageInput

    window = window_at(tmp_path)
    wait_for(app, lambda: window.controller.idle)
    image = ImageInput.from_camera(np.full((180, 180, 3), 120, dtype=np.uint8))
    window.controller.library_operation("create", name="测试", inputs=(image,))
    wait_for(app, lambda: window.controller.idle)
    person = window.controller.catalog.people[0]

    class RollbackFailingRepository(FaceRepository):
        """只拒绝真实 SQLite 回滚，保留正常读取和业务重复检查。"""

        def __init__(self, path):
            """在事务执行前安装故障注入。"""
            super().__init__(path)
            self._connection.set_authorizer(
                lambda action, value, *_: (
                    sqlite3.SQLITE_DENY
                    if action == sqlite3.SQLITE_TRANSACTION and value == "ROLLBACK"
                    else sqlite3.SQLITE_OK
                )
            )

    monkeypatch.setattr(
        "camera_face_comparison.ui.workers.FaceRepository", RollbackFailingRepository
    )
    window.controller.library_operation("append", person_id=person.id, inputs=(image,))
    wait_for(app, lambda: window.controller.idle)
    try:
        assert not window.controller.library_valid
        assert not window.import_compare_button.isEnabled()
        assert "回滚失败" in window._diagnostic_detail
    finally:
        window.close()
        wait_for(app, lambda: window.controller.can_close)


def test_recheck_clears_result_from_previous_library_snapshot(tmp_path, app):
    """另一个连接修改姓名后复查，不能留下原标准库的判定卡片。"""
    from camera_face_comparison.image_input import ImageInput

    window = window_at(tmp_path)
    wait_for(app, lambda: window.controller.idle)
    window.controller.library_operation(
        "create",
        name="原姓名",
        inputs=(ImageInput.from_camera(np.full((180, 180, 3), 120, dtype=np.uint8)),),
    )
    wait_for(app, lambda: window.controller.idle)
    sample = window.controller.catalog.metadata[0]
    window.controller.recognize_file(tmp_path / sample.image_path)
    wait_for(app, lambda: window.controller.idle)
    assert window.result_label.property("state") == "matched"
    repository = FaceRepository(window.controller.settings.database_path)
    repository.rename_person(sample.person_id, "新姓名")
    repository.close()
    window.recheck_library()
    wait_for(app, lambda: window.controller.idle)
    try:
        assert window.controller.catalog.people[0].display_name == "新姓名"
        assert window.result_label.property("state") == "idle"
        assert window.result_preview_label.pixmap().isNull()
    finally:
        window.close()
        wait_for(app, lambda: window.controller.can_close)
