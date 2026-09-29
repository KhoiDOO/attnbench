"""Scaled dot-product attention with a selectable kernel.

Every backend takes and returns tensors in the ``(batch, length, heads, head_dim)`` layout.

Backends:
    - ``"sdpa"``: :func:`torch.nn.functional.scaled_dot_product_attention`. Always available.
    - ``"flash"``: ``flash_attn.flash_attn_func`` (FlashAttention-2). Requires ``pip install flash-attn``.
    - ``"flash4"``: ``flash_attn.cute.flash_attn_func`` (FlashAttention-4). Requires
      ``pip install flash-attn-4``. Forward and backward on compute capability 9.0 and above;
      forward only on 8.x.
    - ``"xformers"``: ``xformers.ops.memory_efficient_attention``. Requires ``pip install xformers``.
    - ``"hffa2"``: FlashAttention-2 from Hugging Face Kernel Hub (``kernels-community/flash-attn2``).
      Requires ``pip install kernels``. Initialized once.
    - ``"auto"``: ``"flash"`` or ``"hffa2"`` when installed and able to handle inputs, otherwise ``"sdpa"``.

Example:
    >>> from conquer3d.nn import attention
    >>> out = attention(q, k, v, backend="auto")
"""

import functools
import importlib
import importlib.metadata
import importlib.util
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

__all__ = [
    "attention",
    "Attention",
    "available_backends",
    "resolve_backend",
    "preload_backend",
    "preload_all",
    "BACKENDS",
    "get_hffa2_kernel",
    "get_hffa3_kernel",
    "get_hffa4_kernel",
]

#: Names accepted by the ``backend`` argument.
BACKENDS = ("auto", "sdpa", "flash", "flash4", "xformers", "xformer", "hffa2", "hffa3", "hffa4")

_PACKAGES = {
    "flash": ("flash_attn", "flash-attn"),
    "flash4": ("flash_attn.cute", "flash-attn-4"),
    "xformers": ("xformers", "xformers"),
    "xformer": ("xformers", "xformers"),
    "hffa2": ("kernels", "kernels"),
    "hffa3": ("kernels", "kernels"),
    "hffa4": ("kernels", "kernels"),
}


def _normalize_window_size(
    window_size: Optional[Union[Tuple[Optional[int], Optional[int]], List[Optional[int]], int]] = None
) -> Tuple[int, int]:
    """Normalizes window_size to a (left, right) integer tuple based on Dao-AILab/flash-attention.

    In FlashAttention, window_size defines the sliding window attention bounds:
    a query at position i attends to keys in [i - window_size[0], i + window_size[1]] inclusive.
    A value of -1 (or None) denotes infinite/unbounded context in that direction.
    An integer w expands to (w, w). Passing None denotes full context attention, mapped to (-1, -1).
    """
    if window_size is None:
        return (-1, -1)
    if isinstance(window_size, int):
        return (window_size, window_size)
    if isinstance(window_size, (tuple, list)) and len(window_size) == 2:
        left = -1 if window_size[0] is None else int(window_size[0])
        right = -1 if window_size[1] is None else int(window_size[1])
        return (left, right)
    raise ValueError(f"window_size must be None, a 2-tuple (left, right), or an int, got {window_size!r}")


def _has_submodule(package: str, name: str) -> bool:
    """Whether `package` ships submodule `name`, found on disk without importing either."""
    try:
        spec = importlib.util.find_spec(package)
    except (ImportError, ValueError):
        return False
    if spec is None or not spec.submodule_search_locations:
        return False
    return any(
        os.path.isfile(os.path.join(loc, name + ".py")) or os.path.isdir(os.path.join(loc, name))
        for loc in spec.submodule_search_locations
    )


def _has_distribution(name: str) -> bool:
    """Whether the pip distribution `name` is installed, without importing it."""
    try:
        importlib.metadata.version(name)
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


# Looked up once, at import, as plain constants: resolution runs inside forward, and
# torch.compile traces constant lookups but not cached or importlib calls. FlashAttention-2 and
# -4 share the `flash_attn` namespace, so each is identified by its own submodule; FlashAttention-2
# wheels also ship an early `cute` directory that cannot run on its own, so -4 additionally needs
# its own distribution to be installed.
_INSTALLED = {
    "sdpa": True,
    "flash": _has_submodule("flash_attn", "flash_attn_interface"),
    "flash4": _has_distribution("flash-attn-4") and _has_submodule("flash_attn", "cute"),
    "xformers": importlib.util.find_spec("xformers") is not None,
    "xformer": importlib.util.find_spec("xformers") is not None,
    "hffa2": importlib.util.find_spec("kernels") is not None,
    "hffa3": importlib.util.find_spec("kernels") is not None,
    "hffa4": importlib.util.find_spec("kernels") is not None,
}


def _installed(backend: str) -> bool:
    return _INSTALLED[backend]


def _import(module: str):
    # A module already in sys.modules is a constant to torch.compile; importing inside a
    # compiled region is not traceable, which is why preload_backend exists.
    loaded = sys.modules.get(module)
    return loaded if loaded is not None else importlib.import_module(module)


_MODULES = {
    "flash": "flash_attn",
    "flash4": "flash_attn.cute",
    "xformers": "xformers.ops",
    "xformer": "xformers.ops",
    "hffa2": "kernels",
    "hffa3": "kernels",
    "hffa4": "kernels",
}

_FLASH_FUNC: Optional[Callable] = None
_FLASH4_FUNC: Optional[Callable] = None
_XFORMERS_OPS: Optional[Any] = None
_XFORMERS_ATTN_FUNC: Optional[Callable] = None
_XFORMERS_MASK: Optional[Any] = None
_HFFA2_KERNEL: Optional[Any] = None
_HFFA2_FUNC: Optional[Callable] = None
_HFFA3_KERNEL: Optional[Any] = None
_HFFA3_FUNC: Optional[Callable] = None
_HFFA4_KERNEL: Optional[Any] = None
_HFFA4_FUNC: Optional[Callable] = None


def _get_flash_func() -> Callable:
    """Loads and caches flash_attn.flash_attn_func once."""
    global _FLASH_FUNC
    if _FLASH_FUNC is None:
        _FLASH_FUNC = _import("flash_attn").flash_attn_func
    return _FLASH_FUNC


def _get_flash4_func() -> Callable:
    """Loads and caches flash_attn.cute.flash_attn_func once."""
    global _FLASH4_FUNC
    if _FLASH4_FUNC is None:
        _FLASH4_FUNC = _import("flash_attn.cute").flash_attn_func
    return _FLASH4_FUNC


def _get_xformers_ops():
    """Loads and caches xformers.ops and its attention functions once."""
    global _XFORMERS_OPS, _XFORMERS_ATTN_FUNC, _XFORMERS_MASK
    if _XFORMERS_OPS is None:
        try:
            _XFORMERS_OPS = _import("xformers.ops")
            _XFORMERS_ATTN_FUNC = getattr(_XFORMERS_OPS, "memory_efficient_attention", None)
            _XFORMERS_MASK = getattr(_XFORMERS_OPS, "LowerTriangularMask", None)
        except Exception:
            _XFORMERS_OPS = None
            _XFORMERS_ATTN_FUNC = None
            _XFORMERS_MASK = None
    return _XFORMERS_OPS


def _get_hffa2_kernel():
    """Loads and caches the Hugging Face kernels-community/flash-attn2 kernel once."""
    global _HFFA2_KERNEL, _HFFA2_FUNC
    if _HFFA2_KERNEL is None:
        import kernels

        try:
            _HFFA2_KERNEL = kernels.get_kernel("kernels-community/flash-attn2", version=3)
        except Exception:
            try:
                _HFFA2_KERNEL = kernels.get_kernel("kernels-community/flash-attn2")
            except Exception:
                _HFFA2_KERNEL = kernels.get_kernel("kernels-community/flash-attn2", version=1)
        _HFFA2_FUNC = _HFFA2_KERNEL.flash_attn_func
    return _HFFA2_KERNEL


def _get_hffa2_func() -> Callable:
    """Loads and caches the Hugging Face flash-attn2 kernel function once."""
    global _HFFA2_FUNC
    if _HFFA2_FUNC is None:
        _get_hffa2_kernel()
    return _HFFA2_FUNC


def get_hffa2_kernel():
    """Returns the cached Hugging Face flash-attn2 kernel module, initializing it if needed."""
    return _get_hffa2_kernel()


def _get_hffa3_kernel():
    """Loads and caches the Hugging Face kernels-community/flash-attn3 kernel once."""
    global _HFFA3_KERNEL, _HFFA3_FUNC
    if _HFFA3_KERNEL is None:
        import kernels

        try:
            _HFFA3_KERNEL = kernels.get_kernel("kernels-community/flash-attn3", version=2, check_arch=False)
        except Exception:
            _HFFA3_KERNEL = kernels.get_kernel("kernels-community/flash-attn3", version=1, check_arch=False)
        _HFFA3_FUNC = getattr(_HFFA3_KERNEL, "flash_attn_func", None)
    return _HFFA3_KERNEL


def _get_hffa3_func() -> Callable:
    """Returns the cached Hugging Face flash-attn3 kernel function once."""
    global _HFFA3_FUNC
    if _HFFA3_FUNC is None:
        _get_hffa3_kernel()
    return _HFFA3_FUNC


def get_hffa3_kernel():
    """Returns the cached Hugging Face flash-attn3 kernel module, initializing it if needed."""
    return _get_hffa3_kernel()


def _get_hffa4_kernel():
    """Loads and caches the Hugging Face kernels-community/flash-attn4 kernel once."""
    global _HFFA4_KERNEL, _HFFA4_FUNC
    if _HFFA4_KERNEL is None:
        try:
            import cutlass.cute as cute
            if not hasattr(cute.core, "ThrMma") and hasattr(cute, "ThrMma"):
                cute.core.ThrMma = cute.ThrMma
            if not hasattr(cute.core, "ThrCopy") and hasattr(cute, "ThrCopy"):
                cute.core.ThrCopy = cute.ThrCopy
            if not hasattr(cute, "make_fragment") and hasattr(cute, "make_rmem_tensor"):
                cute.make_fragment = cute.make_rmem_tensor
        except Exception:
            pass

        import kernels

        try:
            _HFFA4_KERNEL = kernels.get_kernel("kernels-community/flash-attn4", version=0)
        except Exception as exc:
            try:
                _HFFA4_KERNEL = kernels.get_kernel("kernels-community/flash-attn4", revision="main")
            except Exception as exc2:
                raise ImportError(
                    f"kernels-community/flash-attn4 failed to load: {exc}"
                ) from exc
        _HFFA4_FUNC = getattr(_HFFA4_KERNEL, "flash_attn_func", None) or getattr(_HFFA4_KERNEL, "flash_attn", None)
    return _HFFA4_KERNEL


def _get_hffa4_func() -> Callable:
    """Returns the cached Hugging Face flash-attn4 kernel function once."""
    global _HFFA4_FUNC
    if _HFFA4_FUNC is None:
        _get_hffa4_kernel()
    return _HFFA4_FUNC


def get_hffa4_kernel():
    """Returns the cached Hugging Face flash-attn4 kernel module, initializing it if needed."""
    return _get_hffa4_kernel()


def preload_backend(backend: str) -> None:
    """Imports the kernel package that `backend` may dispatch to, if it is installed.

    Call it before :func:`torch.compile` when `backend` is ``"auto"``, ``"flash"``, ``"flash4"``,
    ``"hffa2"``, ``"hffa3"``, ``"hffa4"``, ``"xformers"``, or ``"all"``, so the import does not happen inside the compiled region.
    :class:`~conquer3d.nn.Attention3DBlock` does this itself.

    Args:
        backend (str): One of :data:`BACKENDS` or ``"all"``.

    Example:
        >>> preload_backend("auto")
        >>> preload_backend("all")
    """
    if backend in ("auto", "all"):
        names = ("flash", "flash4", "xformers", "hffa2", "hffa3", "hffa4") if backend == "all" else ("flash",)
    else:
        names = (backend,)
    for name in names:
        try:
            if name in ("xformers", "xformer"):
                if _installed("xformers"):
                    _get_xformers_ops()
            elif name == "flash":
                if _installed("flash"):
                    _get_flash_func()
            elif name == "flash4":
                if _installed("flash4"):
                    _get_flash4_func()
            elif name == "hffa2":
                if _installed("hffa2"):
                    _get_hffa2_kernel()
            elif name == "hffa3":
                if _installed("hffa3"):
                    _get_hffa3_kernel()
            elif name == "hffa4":
                if _installed("hffa4"):
                    _get_hffa4_kernel()
            elif name in _MODULES and _installed(name):
                _import(_MODULES[name])
        except Exception:
            pass


def preload_all() -> None:
    """Eagerly imports and caches all installed attention kernel backends."""
    preload_backend("all")


def available_backends() -> List[str]:
    """Returns the backends whose packages are installed.

    Returns:
        List[str]: Backend names, always including ``"sdpa"``.

    Example:
        >>> available_backends()
        ['sdpa', 'flash', 'hffa2', 'hffa3', 'hffa4']
    """
    return [b for b in ("sdpa", "flash", "flash4", "xformers", "hffa2", "hffa3", "hffa4") if _installed(b)]


def _flash_problem(device_type: str, capability: Tuple[int, int], dtype: torch.dtype, head_dim: int) -> Optional[str]:
    if device_type != "cuda":
        return "flash-attn requires CUDA tensors"
    if dtype not in (torch.float16, torch.bfloat16):
        return f"flash-attn requires float16 or bfloat16, got {dtype}"
    if capability < (8, 0):
        return f"flash-attn requires compute capability 8.0 or higher, got {capability[0]}.{capability[1]}"
    if head_dim > 256 or head_dim % 8 != 0:
        return f"flash-attn requires head_dim <= 256 and divisible by 8, got {head_dim}"
    return None


def _flash4_problem(
    device_type: str,
    capability: Tuple[int, int],
    dtype: torch.dtype,
    head_dim: int,
    needs_grad: bool,
    dropout: bool,
) -> Optional[str]:
    if device_type != "cuda":
        return "flash-attn-4 requires CUDA tensors"
    if dtype not in (torch.float16, torch.bfloat16):
        return f"flash-attn-4 requires float16 or bfloat16, got {dtype}"
    if capability[0] not in (8, 9, 10, 11, 12):
        return f"flash-attn-4 requires compute capability 8.x to 12.x, got {capability[0]}.{capability[1]}"
    if needs_grad and capability[0] == 8:
        return (
            f"flash-attn-4 has no backward pass on compute capability {capability[0]}.{capability[1]}; "
            "run it without gradients (torch.no_grad or inference) or use backend='flash' or 'sdpa'"
        )
    if dropout:
        return "flash-attn-4 does not support attention dropout"
    if head_dim > 256 or head_dim % 8 != 0:
        return f"flash-attn-4 requires head_dim <= 256 and divisible by 8, got {head_dim}"
    return None


def _xformers_problem(device_type: str) -> Optional[str]:
    if device_type != "cuda":
        return "xformers memory_efficient_attention requires CUDA tensors"
    return None


def _hffa2_problem(device_type: str, capability: Tuple[int, int], dtype: torch.dtype, head_dim: int) -> Optional[str]:
    if device_type != "cuda":
        return "hffa2 requires CUDA tensors"
    if dtype not in (torch.float16, torch.bfloat16):
        return f"hffa2 requires float16 or bfloat16, got {dtype}"
    if capability < (8, 0):
        return f"hffa2 requires compute capability 8.0 or higher, got {capability[0]}.{capability[1]}"
    if head_dim > 256 or head_dim % 8 != 0:
        return f"hffa2 requires head_dim <= 256 and divisible by 8, got {head_dim}"
    return None


def _hffa3_problem(device_type: str, capability: Tuple[int, int], dtype: torch.dtype, head_dim: int) -> Optional[str]:
    if device_type != "cuda":
        return "hffa3 requires CUDA tensors"
    if dtype not in (torch.float16, torch.bfloat16):
        return f"hffa3 requires float16 or bfloat16, got {dtype}"
    if capability[0] != 9:
        return f"hffa3 requires Hopper compute capability 9.x, got {capability[0]}.{capability[1]}"
    if head_dim > 256 or head_dim % 8 != 0:
        return f"hffa3 requires head_dim <= 256 and divisible by 8, got {head_dim}"
    return None


def _hffa4_problem(device_type: str, capability: Tuple[int, int], dtype: torch.dtype, head_dim: int) -> Optional[str]:
    if device_type != "cuda":
        return "hffa4 requires CUDA tensors"
    if dtype not in (torch.float16, torch.bfloat16):
        return f"hffa4 requires float16 or bfloat16, got {dtype}"
    if capability[0] not in (8, 9, 10, 11, 12):
        return f"hffa4 requires compute capability 8.x to 12.x, got {capability[0]}.{capability[1]}"
    if head_dim > 256 or head_dim % 8 != 0:
        return f"hffa4 requires head_dim <= 256 and divisible by 8, got {head_dim}"
    return None


def _resolve(
    backend: str,
    device_type: str,
    capability: Tuple[int, int],
    dtype: torch.dtype,
    head_dim: int,
    needs_grad: bool = False,
    dropout: bool = False,
    window_size: Optional[Union[Tuple[Optional[int], Optional[int]], List[Optional[int]], int]] = None,
) -> str:
    if backend not in BACKENDS:
        raise ValueError(f"Unknown attention backend {backend!r}; expected one of {BACKENDS}.")
    if backend == "auto":
        if _installed("flash") and _flash_problem(device_type, capability, dtype, head_dim) is None:
            return "flash"
        if _installed("hffa2") and _hffa2_problem(device_type, capability, dtype, head_dim) is None:
            return "hffa2"
        if window_size is not None:
            if (
                _installed("flash4")
                and _flash4_problem(device_type, capability, dtype, head_dim, needs_grad, dropout) is None
            ):
                return "flash4"
            raise ValueError(
                f"Attention backend 'auto' cannot support window_size={window_size!r} because "
                "neither 'flash', 'hffa2', nor 'flash4' is installed/compatible, and 'sdpa' does not support window_size."
            )
        return "sdpa"
    if backend == "sdpa":
        if window_size is not None:
            raise ValueError(
                f"Attention backend 'sdpa' does not support window_size (expected None, got {window_size!r})."
            )
        return "sdpa"
    if not _installed(backend):
        module, pip_name = _PACKAGES[backend]
        raise ImportError(
            f"Attention backend {backend!r} needs the '{module}' package, which is not installed. "
            f"Install it with `pip install {pip_name}` or `pip install conquer3d[{backend}]`, "
            f"or use backend='sdpa' or 'auto'."
        )
    if backend == "flash":
        problem = _flash_problem(device_type, capability, dtype, head_dim)
    elif backend == "flash4":
        problem = _flash4_problem(device_type, capability, dtype, head_dim, needs_grad, dropout)
    elif backend == "hffa2":
        problem = _hffa2_problem(device_type, capability, dtype, head_dim)
    elif backend == "hffa3":
        problem = _hffa3_problem(device_type, capability, dtype, head_dim)
    elif backend == "hffa4":
        problem = _hffa4_problem(device_type, capability, dtype, head_dim)
    else:
        if window_size is not None:
            raise ValueError(
                f"Attention backend {backend!r} does not support window_size (expected None, got {window_size!r})."
            )
        problem = _xformers_problem(device_type)
    if problem is not None:
        raise ValueError(f"Attention backend {backend!r} cannot run these inputs: {problem}.")
    return backend


def _capability(device: torch.device) -> Tuple[int, int]:
    if device.type != "cuda":
        return (0, 0)
    return torch.cuda.get_device_capability(device)


def resolve_backend(
    backend: str,
    q: torch.Tensor,
    needs_grad: bool = False,
    dropout: bool = False,
    window_size: Optional[Union[Tuple[Optional[int], Optional[int]], List[Optional[int]], int]] = None,
) -> str:
    """Returns the concrete backend that ``attention`` would use for ``q``.

    Args:
        backend (str): One of :data:`BACKENDS`.
        q (torch.Tensor): Query tensor of shape `(B, L, H, D)`.
        needs_grad (bool, optional): Whether a backward pass will run. Defaults to False.
        dropout (bool, optional): Whether attention dropout is non-zero. Defaults to False.
        window_size (Tuple[int, int], int, optional): Sliding window attention ``(left, right)``
            based on Dao-AILab/flash-attention. Defaults to ``None`` (full attention).

    Returns:
        str: ``"sdpa"``, ``"flash"``, ``"flash4"``, ``"xformers"``, ``"hffa2"``, ``"hffa3"``, or ``"hffa4"``.

    Raises:
        ValueError: If `backend` is unknown, cannot handle `q`, or does not support `window_size`.
        ImportError: If `backend` is requested explicitly and its package is not installed.

    Example:
        >>> resolve_backend("auto", q)
        'sdpa'
    """
    return _resolve(
        backend,
        q.device.type,
        _capability(q.device),
        q.dtype,
        q.shape[-1],
        needs_grad,
        dropout,
        window_size,
    )


def _run_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    dropout_p: float,
    scale: Optional[float],
    window_size: Optional[Union[Tuple[Optional[int], Optional[int]], List[Optional[int]], int]] = None,
) -> torch.Tensor:
    if window_size is not None:
        raise ValueError(
            f"Attention backend 'sdpa' does not support window_size (expected None, got {window_size!r})."
        )
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), dropout_p=dropout_p, is_causal=causal, scale=scale
    )
    return out.transpose(1, 2)


def _run_flash(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    dropout_p: float,
    scale: Optional[float],
    window_size: Optional[Union[Tuple[Optional[int], Optional[int]], List[Optional[int]], int]] = None,
) -> torch.Tensor:
    fn = _FLASH_FUNC
    if fn is None:
        fn = _get_flash_func()
    actual_window_size = _normalize_window_size(window_size)
    return fn(
        q, k, v, dropout_p=dropout_p, softmax_scale=scale, causal=causal, window_size=actual_window_size
    )


def _run_flash4(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    dropout_p: float,
    scale: Optional[float],
    window_size: Optional[Union[Tuple[Optional[int], Optional[int]], List[Optional[int]], int]] = None,
) -> torch.Tensor:
    fn = _FLASH4_FUNC
    if fn is None:
        fn = _get_flash4_func()
    actual_window_size = _normalize_window_size(window_size)
    out = fn(
        q, k, v, softmax_scale=scale, causal=causal, window_size=actual_window_size
    )
    return out[0] if isinstance(out, tuple) else out


def _run_xformers(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    dropout_p: float,
    scale: Optional[float],
    window_size: Optional[Union[Tuple[Optional[int], Optional[int]], List[Optional[int]], int]] = None,
) -> torch.Tensor:
    if window_size is not None:
        raise ValueError(
            f"Attention backend 'xformers' does not support window_size (expected None, got {window_size!r})."
        )
    fn = _XFORMERS_ATTN_FUNC
    mask_cls = _XFORMERS_MASK
    if fn is None or mask_cls is None:
        _get_xformers_ops()
        fn = _XFORMERS_ATTN_FUNC
        mask_cls = _XFORMERS_MASK
    if fn is None:
        raise RuntimeError(
            "xformers is installed but xformers.ops.memory_efficient_attention is unavailable (mslk package may be missing)."
        )
    bias = mask_cls() if causal and mask_cls is not None else None
    return fn(q, k, v, attn_bias=bias, p=dropout_p, scale=scale)


def _run_hffa2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    dropout_p: float,
    scale: Optional[float],
    window_size: Optional[Union[Tuple[Optional[int], Optional[int]], List[Optional[int]], int]] = None,
) -> torch.Tensor:
    fn = _HFFA2_FUNC
    if fn is None:
        fn = _get_hffa2_func()
    actual_window_size = _normalize_window_size(window_size)
    return fn(
        q, k, v, dropout_p=dropout_p, softmax_scale=scale, causal=causal, window_size=actual_window_size
    )


def _run_hffa3(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    dropout_p: float,
    scale: Optional[float],
    window_size: Optional[Union[Tuple[Optional[int], Optional[int]], List[Optional[int]], int]] = None,
) -> torch.Tensor:
    cap = _capability(q.device)
    if cap[0] != 9:
        raise RuntimeError(
            f"kernels-community/flash-attn3 requires NVIDIA Hopper architecture (compute capability 9.x), but device is {q.device} (capability {cap[0]}.{cap[1]})."
        )
    fn = _HFFA3_FUNC
    if fn is None:
        fn = _get_hffa3_func()
    if fn is None:
        raise RuntimeError("kernels-community/flash-attn3 failed to load flash_attn_func.")
    actual_window_size = _normalize_window_size(window_size)
    try:
        out = fn(q, k, v, softmax_scale=scale, causal=causal, window_size=actual_window_size)
    except TypeError:
        out = fn(q, k, v, softmax_scale=scale, causal=causal)
    return out[0] if isinstance(out, tuple) else out


def _run_hffa4(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    dropout_p: float,
    scale: Optional[float],
    window_size: Optional[Union[Tuple[Optional[int], Optional[int]], List[Optional[int]], int]] = None,
) -> torch.Tensor:
    fn = _HFFA4_FUNC
    if fn is None:
        fn = _get_hffa4_func()
    if fn is None:
        raise RuntimeError("kernels-community/flash-attn4 failed to load flash_attn_func.")
    actual_window_size = _normalize_window_size(window_size)
    out = fn(q, k, v, softmax_scale=scale, causal=causal, window_size=actual_window_size)
    return out[0] if isinstance(out, tuple) else out


_RUNNERS: Dict[str, Callable] = {
    "sdpa": _run_sdpa,
    "flash": _run_flash,
    "flash4": _run_flash4,
    "xformers": _run_xformers,
    "xformer": _run_xformers,
    "hffa2": _run_hffa2,
    "hffa3": _run_hffa3,
    "hffa4": _run_hffa4,
}


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    backend: str = "auto",
    causal: bool = False,
    dropout_p: float = 0.0,
    scale: Optional[float] = None,
    window_size: Optional[Union[Tuple[Optional[int], Optional[int]], List[Optional[int]], int]] = None,
    skip_check: bool = False,
) -> torch.Tensor:
    """Scaled dot-product attention over `(B, L, H, D)` tensors with a selectable kernel.

    Args:
        q (torch.Tensor): Queries of shape `(B, Lq, H, D)`.
        k (torch.Tensor): Keys of shape `(B, Lk, H, D)`.
        v (torch.Tensor): Values of shape `(B, Lk, H, D)`.
        backend (str, optional): One of :data:`BACKENDS`. Defaults to ``"auto"``.
        causal (bool, optional): Whether to apply a causal mask. Defaults to False.
        dropout_p (float, optional): Attention dropout probability. Defaults to 0.0.
        scale (float, optional): Softmax scale. Defaults to ``1 / sqrt(D)``.
        window_size (Tuple[int, int] or int, optional): Sliding window attention
            ``(window_size_left, window_size_right)`` based on Dao-AILab/flash-attention.
            If specified, a query at position ``i`` will only attend to keys
            between ``[i - window_size[0], i + window_size[1]]`` (inclusive).
            Passing ``-1`` or ``None`` indicates infinite context in that direction.
            An integer ``w`` expands to ``(w, w)``. Defaults to ``None`` (full context).
            Supported by ``"flash"``, ``"flash4"``, ``"hffa2"``, ``"hffa3"``, and ``"hffa4"``; raises ValueError if not None
            for ``"sdpa"`` or ``"xformers"``.
        skip_check (bool, optional): If True and ``backend`` is not ``"auto"``, skips
            ``resolve_backend`` input validation and capability checks. Defaults to False.

    Returns:
        torch.Tensor: Output of shape `(B, Lq, H, D)`.

    Raises:
        ValueError: If `backend` is unknown, cannot handle the inputs, or does not support `window_size`.
        ImportError: If `backend` is requested explicitly and its package is not installed.

    Example:
        >>> out = attention(q, k, v, backend="flash", window_size=(512, 0), causal=True)
    """
    if backend not in BACKENDS:
        raise ValueError(f"Unknown attention backend {backend!r}; expected one of {BACKENDS}.")
    if backend in ("sdpa", "xformers", "xformer") and window_size is not None:
        raise ValueError(
            f"Attention backend {backend!r} does not support window_size (expected None, got {window_size!r})."
        )
    if backend != "auto" and skip_check:
        if backend == "flash":
            fn = _FLASH_FUNC if _FLASH_FUNC is not None else _get_flash_func()
            return fn(
                q,
                k,
                v,
                dropout_p=dropout_p,
                softmax_scale=scale,
                causal=causal,
                window_size=_normalize_window_size(window_size),
            )
        elif backend == "sdpa":
            out = F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), dropout_p=dropout_p, is_causal=causal, scale=scale
            )
            return out.transpose(1, 2)
        elif backend == "hffa2":
            fn = _HFFA2_FUNC if _HFFA2_FUNC is not None else _get_hffa2_func()
            return fn(
                q,
                k,
                v,
                dropout_p=dropout_p,
                softmax_scale=scale,
                causal=causal,
                window_size=_normalize_window_size(window_size),
            )
        elif backend == "hffa3":
            cap = _capability(q.device)
            if cap[0] != 9:
                raise RuntimeError(
                    f"kernels-community/flash-attn3 requires NVIDIA Hopper architecture (compute capability 9.x), but device is {q.device} (capability {cap[0]}.{cap[1]})."
                )
            fn = _HFFA3_FUNC if _HFFA3_FUNC is not None else _get_hffa3_func()
            if fn is None:
                raise RuntimeError("kernels-community/flash-attn3 failed to load flash_attn_func.")
            try:
                out = fn(q, k, v, softmax_scale=scale, causal=causal, window_size=_normalize_window_size(window_size))
            except TypeError:
                out = fn(q, k, v, softmax_scale=scale, causal=causal)
            return out[0] if isinstance(out, tuple) else out
        elif backend == "hffa4":
            fn = _HFFA4_FUNC if _HFFA4_FUNC is not None else _get_hffa4_func()
            if fn is None:
                raise RuntimeError("kernels-community/flash-attn4 failed to load flash_attn_func.")
            out = fn(q, k, v, softmax_scale=scale, causal=causal, window_size=_normalize_window_size(window_size))
            return out[0] if isinstance(out, tuple) else out
        elif backend in ("xformers", "xformer"):
            fn = _XFORMERS_ATTN_FUNC
            mask_cls = _XFORMERS_MASK
            if fn is None or mask_cls is None:
                _get_xformers_ops()
                fn = _XFORMERS_ATTN_FUNC
                mask_cls = _XFORMERS_MASK
            if fn is None:
                return _run_xformers(q, k, v, causal, dropout_p, scale, window_size)
            bias = mask_cls() if (causal and mask_cls is not None) else None
            return fn(q, k, v, attn_bias=bias, p=dropout_p, scale=scale)
        elif backend == "flash4":
            fn = _FLASH4_FUNC if _FLASH4_FUNC is not None else _get_flash4_func()
            out = fn(
                q,
                k,
                v,
                softmax_scale=scale,
                causal=causal,
                window_size=_normalize_window_size(window_size),
            )
            return out[0] if isinstance(out, tuple) else out
        name = backend
    elif backend == "sdpa":
        name = "sdpa"
    else:
        needs_grad = torch.is_grad_enabled() and (q.requires_grad or k.requires_grad or v.requires_grad)
        name = resolve_backend(backend, q, needs_grad, dropout_p > 0.0, window_size)
    return _RUNNERS[name](q, k, v, causal, dropout_p, scale, window_size)


class Attention(nn.Module):
    """A :class:`torch.nn.Module` wrapper around :func:`attention`.

    Args:
        backend (str, optional): One of :data:`BACKENDS`. Defaults to ``"auto"``.
        causal (bool, optional): Whether to apply a causal mask. Defaults to False.
        dropout_p (float, optional): Attention dropout probability. Defaults to 0.0.
        scale (float, optional): Softmax scale. Defaults to ``1 / sqrt(D)``.
        window_size (Tuple[int, int] or int, optional): Sliding window attention
            ``(window_size_left, window_size_right)`` based on Dao-AILab/flash-attention.
            If specified, a query at position ``i`` will only attend to keys
            between ``[i - window_size[0], i + window_size[1]]`` (inclusive).
            Passing ``-1`` or ``None`` indicates infinite context in that direction.
            An integer ``w`` expands to ``(w, w)``. Defaults to ``None`` (full context).
            Supported by ``"flash"``, ``"flash4"``, ``"hffa2"``, ``"hffa3"``, and ``"hffa4"``; raises ValueError if not None
            for ``"sdpa"`` or ``"xformers"``.
            skip_check (bool, optional): If True and ``backend`` is not ``"auto"``, skips
            ``resolve_backend`` input validation and capability checks. Defaults to False.

    Example:
        >>> attn = Attention(backend="flash", causal=True)
        >>> out = attn(q, k, v)
    """

    def __init__(
        self,
        backend: str = "auto",
        causal: bool = False,
        dropout_p: float = 0.0,
        scale: Optional[float] = None,
        window_size: Optional[Union[Tuple[Optional[int], Optional[int]], List[Optional[int]], int]] = None,
        skip_check: bool = False,
    ):
        super().__init__()
        if backend not in BACKENDS:
            raise ValueError(f"Unknown attention backend {backend!r}; expected one of {BACKENDS}.")
        if backend in ("sdpa", "xformers", "xformer") and window_size is not None:
            raise ValueError(
                f"Attention backend {backend!r} does not support window_size (expected None, got {window_size!r})."
            )
        self.backend = backend
        self.causal = causal
        self.dropout_p = dropout_p
        self.scale = scale
        self.window_size = window_size
        self.skip_check = skip_check
        if backend != "auto":
            if not _installed(backend):
                module, pip_name = _PACKAGES[backend]
                raise ImportError(
                    f"Attention backend {backend!r} needs the '{module}' package, which is not installed. "
                    f"Install it with `pip install {pip_name}` or `pip install conquer3d[{backend}]`, "
                    f"or use backend='sdpa' or 'auto'."
                )
            preload_backend(backend)
            self._runner: Optional[Callable] = _RUNNERS[backend]

            actual_window = _normalize_window_size(window_size)
            if backend == "flash":
                fn = _FLASH_FUNC if _FLASH_FUNC is not None else _get_flash_func()
                self._forward_impl: Optional[Callable] = lambda q, k, v: fn(
                    q, k, v, dropout_p=dropout_p, softmax_scale=scale, causal=causal, window_size=actual_window
                )
            elif backend == "sdpa":
                self._forward_impl = lambda q, k, v: F.scaled_dot_product_attention(
                    q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), dropout_p=dropout_p, is_causal=causal, scale=scale
                ).transpose(1, 2)
            elif backend == "hffa2":
                fn = _HFFA2_FUNC if _HFFA2_FUNC is not None else _get_hffa2_func()
                self._forward_impl = lambda q, k, v: fn(
                    q, k, v, dropout_p=dropout_p, softmax_scale=scale, causal=causal, window_size=actual_window
                )
            elif backend == "hffa3":
                fn = _HFFA3_FUNC if _HFFA3_FUNC is not None else _get_hffa3_func()

                def _hffa3_runner(q, k, v):
                    cap = _capability(q.device)
                    if cap[0] != 9:
                        raise RuntimeError(
                            f"kernels-community/flash-attn3 requires NVIDIA Hopper architecture (compute capability 9.x), but device is {q.device} (capability {cap[0]}.{cap[1]})."
                        )
                    if fn is None:
                        raise RuntimeError("kernels-community/flash-attn3 failed to load flash_attn_func.")
                    try:
                        out = fn(q, k, v, softmax_scale=scale, causal=causal, window_size=actual_window)
                    except TypeError:
                        out = fn(q, k, v, softmax_scale=scale, causal=causal)
                    return out[0] if isinstance(out, tuple) else out

                self._forward_impl = _hffa3_runner
            elif backend == "hffa4":
                fn = _HFFA4_FUNC if _HFFA4_FUNC is not None else _get_hffa4_func()

                def _hffa4_runner(q, k, v):
                    if fn is None:
                        raise RuntimeError("kernels-community/flash-attn4 failed to load flash_attn_func.")
                    out = fn(q, k, v, softmax_scale=scale, causal=causal, window_size=actual_window)
                    return out[0] if isinstance(out, tuple) else out

                self._forward_impl = _hffa4_runner
            elif backend in ("xformers", "xformer"):
                fn = _XFORMERS_ATTN_FUNC
                mask_cls = _XFORMERS_MASK
                bias = mask_cls() if (causal and mask_cls is not None) else None
                if fn is None:
                    self._forward_impl = lambda q, k, v: self._runner(q, k, v, causal, dropout_p, scale, window_size)
                else:
                    self._forward_impl = lambda q, k, v: fn(q, k, v, attn_bias=bias, p=dropout_p, scale=scale)
            elif backend == "flash4":
                fn = _FLASH4_FUNC if _FLASH4_FUNC is not None else _get_flash4_func()

                def _flash4_runner(q, k, v):
                    out = fn(q, k, v, softmax_scale=scale, causal=causal, window_size=actual_window)
                    return out[0] if isinstance(out, tuple) else out

                self._forward_impl = _flash4_runner
            else:
                self._forward_impl = lambda q, k, v: self._runner(q, k, v, causal, dropout_p, scale, window_size)
        else:
            self._runner = None
            self._forward_impl = None

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if self.skip_check and self._forward_impl is not None:
            return self._forward_impl(q, k, v)
        return attention(
            q,
            k,
            v,
            backend=self.backend,
            causal=self.causal,
            dropout_p=self.dropout_p,
            scale=self.scale,
            window_size=self.window_size,
            skip_check=self.skip_check,
        )