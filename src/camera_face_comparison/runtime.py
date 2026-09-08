from __future__ import annotations

import ctypes
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class ExecutionBackend:
    """当前 ONNX Runtime 推理后端及其 InsightFace 上下文参数。"""

    name: Literal["cuda", "cpu"]
    providers: tuple[str, ...]
    context_id: int


def detect_execution_backend() -> ExecutionBackend:
    """根据当前安装的 ONNX Runtime provider 选择 CUDA 或 CPU。

    返回：
        CUDA 可用时返回 CUDA 优先且带 CPU 回退的后端，否则返回 CPU 后端。
    前置条件：
        当前 Python 环境已安装 ONNX Runtime；CUDA provider 还必须能加载对应驱动。
    """

    ort = _load_onnxruntime()
    return select_execution_backend(tuple(ort.get_available_providers()))


def actual_backend_for_analyzer(analyzer: object, requested: ExecutionBackend) -> ExecutionBackend:
    """根据 InsightFace 已创建的 session 检查实际使用的 provider。

    参数：
        analyzer：已执行 `prepare` 的 InsightFace `FaceAnalysis` 对象。
        requested：创建分析器时请求的后端。
    返回：
        所有可检查模型都使用 CUDA 时返回原请求；发现实际回退到 CPU 时返回 CPU。
    前置条件：
        分析器已经完成模型加载。无法检查 session 的测试替身会保留请求结果。
    """

    if requested.name != "cuda":
        return requested
    models = getattr(analyzer, "models", None)
    if not isinstance(models, dict):
        return requested
    session_providers: list[list[str]] = []
    for model in models.values():
        session = getattr(model, "session", None)
        get_providers = getattr(session, "get_providers", None)
        if callable(get_providers):
            providers = [str(provider) for provider in get_providers()]
            if providers:
                session_providers.append(providers)
    if session_providers and not all(
        providers[0] == "CUDAExecutionProvider" for providers in session_providers
    ):
        return _cpu_backend()
    return requested


def backend_metadata(backend: ExecutionBackend) -> dict[str, object]:
    """把实际推理后端转换为可写入评测报告的 JSON 字段。"""

    return {
        "name": backend.name,
        "providers": list(backend.providers),
        "context_id": backend.context_id,
    }


def select_execution_backend(available_providers: Sequence[str]) -> ExecutionBackend:
    """从 ONNX Runtime 已注册 provider 中选择优先级最高的执行后端。

    参数：
        available_providers：`onnxruntime.get_available_providers()` 的结果。
    返回：
        CUDA provider 存在时使用 CUDA 并保留 CPU 回退；否则使用 CPU。
    前置条件：
        provider 列表至少包含 `CPUExecutionProvider`。
    """

    if "CUDAExecutionProvider" in available_providers:
        return ExecutionBackend(
            name="cuda",
            providers=("CUDAExecutionProvider", "CPUExecutionProvider"),
            context_id=0,
        )
    if "CPUExecutionProvider" in available_providers:
        return _cpu_backend()
    raise RuntimeError("ONNX Runtime has no usable CPU or CUDA execution provider")


def _cpu_backend() -> ExecutionBackend:
    """构造统一的 CPU 后端描述。"""

    return ExecutionBackend(name="cpu", providers=("CPUExecutionProvider",), context_id=-1)


def _load_onnxruntime():
    """导入 ONNX Runtime，并尽量预加载虚拟环境中的 CUDA 动态库。

    返回：
        已导入的 `onnxruntime` 模块。
    前置条件：
        未安装 ONNX Runtime 时抛出面向用户的运行时错误。
    """

    try:
        import onnxruntime as ort
    except ImportError as error:
        raise RuntimeError("ONNX Runtime is not installed; install project dependencies first") from error
    _preload_cuda_libraries(ort)
    return ort


def _preload_cuda_libraries(ort: object) -> None:
    """从当前 Python 环境加载 CUDA 13/cuDNN 共享库，避免依赖 shell 环境变量。"""

    package_root = _site_package_root()
    nvidia_root = package_root / "nvidia"
    if not nvidia_root.is_dir():
        preload = getattr(ort, "preload_dlls", None)
        if callable(preload):
            preload()
        return

    library_names = (
        "libcudart.so.13",
        "libcublas.so.13",
        "libcublasLt.so.13",
        "libcurand.so.10",
    )
    library_paths = {
        name: path
        for path in sorted(nvidia_root.glob("*/lib/*"))
        for name in library_names
        if path.name == name
    }
    for name in library_names:
        path = library_paths.get(name)
        if path is not None:
            _load_shared_library(path)

    # cuDNN 9 拆分为多个共享库，先加载主库及其组件，失败时交给 ORT 自己报错。
    for path in sorted(nvidia_root.glob("*/lib/libcudnn*.so.*")):
        _load_shared_library(path)


def _load_shared_library(path: Path) -> None:
    """以全局符号方式加载单个动态库，已加载或不可用时继续启动。"""

    try:
        ctypes.CDLL(str(path), mode=getattr(ctypes, "RTLD_GLOBAL", 0))
    except OSError:
        pass


def _site_package_root() -> Path:
    """返回当前解释器的 site-packages 根目录。"""

    executable = Path(sys.executable).resolve()
    candidates = [
        executable.parent.parent / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return Path(__file__).resolve().parents[2]
