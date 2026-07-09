from __future__ import annotations

from typing import Literal

Backend = Literal["numpy", "numba"]
LowessBackend = Literal["statsmodels", "numba"]
FisherBackend = Literal["scipy", "numba"]

DEFAULT_BACKEND: Backend = "numba"
DEFAULT_LOWESS_BACKEND: LowessBackend = "statsmodels"
DEFAULT_FISHER_BACKEND: FisherBackend = "scipy"


def validate_backend(backend: str) -> Backend:
    if backend not in {"numpy", "numba"}:
        raise ValueError("backend must be one of: numpy, numba")
    if backend == "numba":
        require_numba()
    return backend  # type: ignore[return-value]


def validate_lowess_backend(lowess_backend: str) -> LowessBackend:
    if lowess_backend not in {"statsmodels", "numba"}:
        raise ValueError("lowess_backend must be one of: statsmodels, numba")
    if lowess_backend == "numba":
        require_numba()
    return lowess_backend  # type: ignore[return-value]


def validate_fisher_backend(fisher_backend: str) -> FisherBackend:
    if fisher_backend not in {"scipy", "numba"}:
        raise ValueError("fisher_backend must be one of: scipy, numba")
    if fisher_backend == "numba":
        require_numba()
    return fisher_backend  # type: ignore[return-value]


def has_numba() -> bool:
    try:
        import numba  # noqa: F401
    except ImportError:
        return False
    return True


def require_numba() -> None:
    if not has_numba():
        raise RuntimeError(
            "Numba backend requested, but numba is not installed. Numba is a "
            "core dependency of ep-touche; if this error occurs, your "
            "installation may be incomplete -- try `uv sync --dev` or "
            "reinstalling `ep-touche`."
        )
