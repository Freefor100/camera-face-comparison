"""选择并核验 InsightFace 使用的 ONNX Runtime 推理后端。"""

from __future__ import annotations

import ctypes
import sysconfig
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class ExecutionBackend:
    """当前 ONNX Runtime 推理后端及其 InsightFace 上下文参数。"""

    name: Literal["cuda", "cpu", "mixed"]
    providers: tuple[str, ...]
    context_id: int
    model_providers: tuple[tuple[str, tuple[str, ...]], ...] = ()


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
    """读取检测和特征会话的实际首选后端，不把混合执行误写成纯 CPU。"""
    models = getattr(analyzer, "models", None)
    if not isinstance(models, dict):
        return requested
    roles = []
    for role in ("detection", "recognition"):
        session = getattr(models.get(role), "session", None)
        getter = getattr(session, "get_providers", None)
        if callable(getter):
            values = tuple(getter())
            if values:
                roles.append((role, values))
    if not roles:
        return requested
    priorities = {providers[0] for _, providers in roles}
    name = (
        "cuda"
        if priorities == {"CUDAExecutionProvider"}
        else "cpu"
        if priorities == {"CPUExecutionProvider"}
        else "mixed"
    )
    providers = tuple(dict.fromkeys(value for _, values in roles for value in values))
    return ExecutionBackend(
        name, providers, -1 if name == "cpu" else requested.context_id, tuple(roles)
    )


def backend_metadata(backend: ExecutionBackend) -> dict[str, object]:
    """把实际推理后端转换为可写入评测报告的 JSON 字段。"""

    return {
        "name": backend.name,
        "providers": list(backend.providers),
        "context_id": backend.context_id,
        "model_providers": {role: list(values) for role, values in backend.model_providers},
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
        # CPU 保留为次级 provider，让不支持 CUDA 的个别算子仍可由 ORT 执行。
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
        raise RuntimeError(
            "ONNX Runtime is not installed; install project dependencies first"
        ) from error
    _preload_cuda_libraries(ort)
    return ort


def _preload_cuda_libraries(ort: object) -> None:
    """从当前 Python 环境加载 CUDA 13/cuDNN 共享库，避免依赖 shell 环境变量。"""

    package_root = _site_package_root()
    nvidia_root = package_root / "nvidia"
    if not nvidia_root.is_dir():
        # 非 pip 拆包布局交回 ORT 自带的加载器处理，避免猜测系统 CUDA 路径。
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
    # 按依赖顺序主动加载运行库，使 ORT 创建 CUDA session 时能解析其全局符号。
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
    """使用当前虚拟环境的安装路径，不解析基础解释器的符号链接。"""
    return Path(sysconfig.get_path("purelib"))
