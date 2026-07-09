"""Backend choices for `lowess_backend`/`fisher_backend`, and dependency checks.

Public API: the `LowessBackend`/`FisherBackend` type aliases, their
`DEFAULT_*` constants, `validate_lowess_backend`/`validate_fisher_backend`
(called from CLI argument parsing and pipeline functions), and the
`has_*`/`require_*` dependency-check pairs. There's no counting `backend`
choice here anymore -- counting always uses numba (see CLAUDE.md) -- these
are only for the lowess smoother and the Fisher exact test, where the numba
path is a validated approximation rather than a bit-exact equivalent.
"""

from __future__ import annotations

from typing import Literal

LowessBackend = Literal["statsmodels", "numba"]
FisherBackend = Literal["scipy", "numba"]

DEFAULT_LOWESS_BACKEND: LowessBackend = "numba"
DEFAULT_FISHER_BACKEND: FisherBackend = "numba"


def validate_lowess_backend(lowess_backend: str) -> LowessBackend:
    """Check `lowess_backend` is a known choice and its dependency is installed."""
    if lowess_backend not in {"statsmodels", "numba"}:
        raise ValueError("lowess_backend must be one of: statsmodels, numba")
    if lowess_backend == "numba":
        require_numba()
    else:
        require_statsmodels()
    return lowess_backend  # type: ignore[return-value]


def validate_fisher_backend(fisher_backend: str) -> FisherBackend:
    """Check `fisher_backend` is a known choice and its dependency is installed."""
    if fisher_backend not in {"scipy", "numba"}:
        raise ValueError("fisher_backend must be one of: scipy, numba")
    if fisher_backend == "numba":
        require_numba()
    return fisher_backend  # type: ignore[return-value]


def has_numba() -> bool:
    """Whether numba is importable. Numba is a core dependency; this should always be true."""
    try:
        import numba  # noqa: F401
    except ImportError:
        return False
    return True


def has_statsmodels() -> bool:
    """Whether statsmodels (part of the optional `legacy` extra) is importable."""
    try:
        import statsmodels  # noqa: F401
    except ImportError:
        return False
    return True


def require_numba() -> None:
    """Raise a clear error if numba is missing, instead of a raw ImportError deep in a call stack."""
    if not has_numba():
        raise RuntimeError(
            "Numba backend requested, but numba is not installed. Numba is a "
            "core dependency of ep-touche; if this error occurs, your "
            "installation may be incomplete -- try `uv sync --dev` or "
            "reinstalling `ep-touche`."
        )


def require_statsmodels() -> None:
    """Raise a clear error pointing at the `legacy` extra if statsmodels is missing."""
    if not has_statsmodels():
        raise RuntimeError(
            "This feature requires statsmodels, which is part of ep-touche's "
            "optional `legacy` extra. Install it with `pip install "
            "ep-touche[legacy]` or `uv sync --extra legacy`."
        )
