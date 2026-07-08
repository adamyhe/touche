from __future__ import annotations

from typing import Literal

Backend = Literal["numpy", "numba"]


def validate_backend(backend: str) -> Backend:
    if backend not in {"numpy", "numba"}:
        raise ValueError("backend must be one of: numpy, numba")
    if backend == "numba":
        require_numba()
    return backend  # type: ignore[return-value]


def has_numba() -> bool:
    try:
        import numba  # noqa: F401
    except ImportError:
        return False
    return True


def require_numba() -> None:
    if not has_numba():
        raise RuntimeError(
            "Numba backend requested, but numba is not installed. "
            "Install the optional speed extra with `pip install 'ep-touche[fast]'` "
            "or `uv sync --extra fast`."
        )
