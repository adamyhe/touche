from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Any


def print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def print_dataclass(value: Any) -> None:
    if not is_dataclass(value):
        raise TypeError(f"Expected dataclass instance, got {type(value)!r}")
    print_json(asdict(value))
