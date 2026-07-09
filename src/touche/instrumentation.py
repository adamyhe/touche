from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, TypeVar

T = TypeVar("T")


@dataclass(slots=True)
class Instrumentation:
    """Optional progress bars and step timings for CLI and notebook workflows."""

    progress: bool = False
    profile: bool = False
    timings: list[dict[str, Any]] = field(default_factory=list)

    def iter(
        self,
        iterable: Iterable[T],
        *,
        total: int | None = None,
        desc: str | None = None,
        unit: str = "it",
        leave: bool = True,
    ) -> Iterable[T]:
        if not self.progress:
            return iterable

        from tqdm.auto import tqdm

        return tqdm(iterable, total=total, desc=desc, unit=unit, leave=leave)

    @contextmanager
    def step(self, name: str) -> Iterator[None]:
        started = perf_counter()
        try:
            yield
        finally:
            if self.profile:
                self.timings.append(
                    {
                        "step": name,
                        "elapsed_seconds": round(perf_counter() - started, 6),
                    }
                )


def make_instrumentation(
    progress: bool | Instrumentation | None = False,
    *,
    profile: bool = False,
) -> Instrumentation:
    if isinstance(progress, Instrumentation):
        if profile:
            progress.profile = True
        return progress
    return Instrumentation(progress=bool(progress), profile=profile)
