from __future__ import annotations

from typing import Literal

LowessBackend = Literal["statsmodels", "numba"]
FisherBackend = Literal["scipy", "numba"]

DEFAULT_LOWESS_BACKEND: LowessBackend = "numba"
DEFAULT_FISHER_BACKEND: FisherBackend = "numba"


def validate_lowess_backend(lowess_backend: str) -> LowessBackend:
    if lowess_backend not in {"statsmodels", "numba"}:
        raise ValueError("lowess_backend must be one of: statsmodels, numba")
    if lowess_backend == "numba":
        require_numba()
    else:
        require_statsmodels()
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


def has_statsmodels() -> bool:
    try:
        import statsmodels  # noqa: F401
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


def require_statsmodels() -> None:
    if not has_statsmodels():
        raise RuntimeError(
            "This feature requires statsmodels, which is part of ep-touche's "
            "optional `legacy` extra. Install it with `pip install "
            "ep-touche[legacy]` or `uv sync --extra legacy`."
        )
