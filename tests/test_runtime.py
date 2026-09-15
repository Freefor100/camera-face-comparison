from __future__ import annotations


def test_execution_backend_prefers_cuda_when_provider_is_available() -> None:
    """存在 CUDA provider 时必须优先使用 GPU，并保留 CPU 回退。"""

    from camera_face_comparison.runtime import select_execution_backend

    backend = select_execution_backend(("CUDAExecutionProvider", "CPUExecutionProvider"))

    assert backend.name == "cuda"
    assert backend.providers == ("CUDAExecutionProvider", "CPUExecutionProvider")
    assert backend.context_id == 0


def test_execution_backend_uses_cpu_when_cuda_is_unavailable() -> None:
    """没有 CUDA provider 时必须稳定使用 CPU。"""

    from camera_face_comparison.runtime import select_execution_backend

    backend = select_execution_backend(("AzureExecutionProvider", "CPUExecutionProvider"))

    assert backend.name == "cpu"
    assert backend.providers == ("CPUExecutionProvider",)
    assert backend.context_id == -1


class _FakeSession:
    """提供模型实际执行后端列表的测试会话。"""

    def __init__(self, providers: tuple[str, ...]) -> None:
        """保存当前测试场景声明的执行后端顺序。"""

        self._providers = providers

    def get_providers(self) -> list[str]:
        """返回测试用的实际 provider 顺序。"""
        return list(self._providers)


class _FakeModel:
    """把测试会话包装为 InsightFace 模型形状。"""

    def __init__(self, providers: tuple[str, ...]) -> None:
        """创建使用指定执行后端的测试模型。"""

        self.session = _FakeSession(providers)


class _FakeAnalyzer:
    """提供检测与识别两个模型的分析器替身。"""

    def __init__(self, providers: tuple[str, ...]) -> None:
        """让两个测试模型使用相同的执行后端列表。"""

        self.models = {
            "detection": _FakeModel(providers),
            "recognition": _FakeModel(providers),
        }


def test_actual_backend_reports_cpu_when_requested_cuda_falls_back() -> None:
    """InsightFace 实际只创建 CPU session 时必须报告 CPU 回退。"""

    from camera_face_comparison.runtime import (
        actual_backend_for_analyzer,
        select_execution_backend,
    )

    requested = select_execution_backend(("CUDAExecutionProvider", "CPUExecutionProvider"))

    backend = actual_backend_for_analyzer(
        _FakeAnalyzer(("CPUExecutionProvider",)), requested
    )

    assert backend.name == "cpu"
    assert backend.providers == ("CPUExecutionProvider",)
    assert backend.context_id == -1


def test_actual_backend_keeps_cuda_when_all_sessions_use_cuda() -> None:
    """InsightFace 的所有可见 session 使用 CUDA 时保留 CUDA 后端。"""

    from camera_face_comparison.runtime import (
        actual_backend_for_analyzer,
        select_execution_backend,
    )

    requested = select_execution_backend(("CUDAExecutionProvider", "CPUExecutionProvider"))

    backend = actual_backend_for_analyzer(
        _FakeAnalyzer(("CUDAExecutionProvider", "CPUExecutionProvider")), requested
    )

    assert backend == requested
