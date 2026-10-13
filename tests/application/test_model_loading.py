"""验证当前虚拟环境定位和真实会话后端的展示契约。"""

import sysconfig
from pathlib import Path
from types import SimpleNamespace

from camera_face_comparison.runtime import (
    ExecutionBackend,
    _site_package_root,
    actual_backend_for_analyzer,
)


def test_cuda_library_location_uses_current_virtual_environment():
    """解释器符号链接不能把依赖定位带到基础环境。"""
    assert _site_package_root() == Path(sysconfig.get_path("purelib"))


def test_mixed_model_sessions_are_not_misreported_as_cpu_only():
    """检测使用 CUDA、特征使用 CPU 时，报告必须保留两种执行事实。"""
    analyzer = SimpleNamespace(
        models={
            "detection": SimpleNamespace(
                session=SimpleNamespace(
                    get_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]
                )
            ),
            "recognition": SimpleNamespace(
                session=SimpleNamespace(get_providers=lambda: ["CPUExecutionProvider"])
            ),
        }
    )
    backend = actual_backend_for_analyzer(
        analyzer, ExecutionBackend("cuda", ("CUDAExecutionProvider", "CPUExecutionProvider"), 0)
    )
    assert backend.name == "mixed"
    assert dict(backend.model_providers)["detection"][0] == "CUDAExecutionProvider"
    assert dict(backend.model_providers)["recognition"][0] == "CPUExecutionProvider"
