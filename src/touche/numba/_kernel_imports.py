"""Shared numba import guard for the `touche.numba.*` kernel modules.

Each kernel module does `from touche.numba._kernel_imports import njit,
prange, get_thread_id` instead of importing `numba` directly, so the
friendly error is defined once.
"""

from __future__ import annotations

try:
    from numba import get_thread_id, njit, prange
except ImportError as exc:  # pragma: no cover - exercised through backend helper
    raise RuntimeError(
        "Numba kernels require numba, which is a core dependency of ep-touche. "
        "If this error occurs, your installation may be incomplete -- try "
        "`uv sync --dev` or reinstalling `ep-touche`."
    ) from exc

__all__ = ["get_thread_id", "njit", "prange"]
