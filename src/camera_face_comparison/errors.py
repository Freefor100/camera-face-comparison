"""把输入、业务、存储和设备异常转换为可跨线程传递的中文错误。"""

import sqlite3
from dataclasses import dataclass


class OperationCancelled(RuntimeError):
    """表示操作者取消尚未提交的工作。"""


class LibraryUnavailableError(RuntimeError):
    """标准库缺失或结构错误，不能作为正常空库使用。"""


@dataclass(frozen=True)
class TaskFailure:
    """保留错误类别、用户说明与诊断详情，而非只发送异常字符串。"""

    kind: str
    message: str
    detail: str


def classify_error(error: Exception, *, scope="system") -> TaskFailure:
    """按实际 SQLite 错误码和操作范围分类，锁竞争不等于损坏。"""
    detail = "；".join([str(error), *getattr(error, "__notes__", [])])
    if isinstance(error, OperationCancelled):
        return TaskFailure("cancelled", "操作已取消。", detail)
    if isinstance(error, sqlite3.Error):
        code = getattr(error, "sqlite_errorcode", 0) & 255
        if code in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}:
            return TaskFailure("storage_busy", "数据库暂时忙碌，请稍后重试。", detail)
        return TaskFailure(
            "storage_unavailable", "数据库无法正常读取或写入，请检查标准库。", detail
        )
    if isinstance(error, LibraryUnavailableError):
        return TaskFailure("storage_unavailable", "标准库无法使用，请查看诊断并重新检查。", detail)
    if scope == "device":
        return TaskFailure("device_error", "摄像头操作失败。", detail)
    if isinstance(error, ValueError):
        return TaskFailure("business_constraint", "输入或操作不符合要求。", detail)
    if isinstance(error, OSError):
        return TaskFailure("file_error", "图片文件无法读取或保存。", detail)
    return TaskFailure("system_error", "处理任务出现异常，请重试或查看诊断。", detail)
