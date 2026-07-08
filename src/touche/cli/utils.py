from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from typing import Any

from touche.instrumentation import Instrumentation


def print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def print_dataclass(value: Any) -> None:
    if not is_dataclass(value):
        raise TypeError(f"Expected dataclass instance, got {type(value)!r}")
    print_json(asdict(value))


def add_instrumentation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show tqdm progress bars on stderr.",
    )
    parser.add_argument(
        "--profile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Collect lightweight step timings in command JSON output.",
    )


def make_cli_instrumentation(args: argparse.Namespace) -> Instrumentation:
    return Instrumentation(
        progress=bool(getattr(args, "progress", False)),
        profile=bool(getattr(args, "profile", False)),
    )


def add_timings(payload: dict[str, Any], instrument: Instrumentation) -> dict[str, Any]:
    if instrument.profile:
        payload["timings"] = instrument.timings
    return payload
