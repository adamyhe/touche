"""Shared helpers used by every `touche.cli.*` command module.

Public API (all of it, since this module is itself CLI-internal plumbing):
`print_json`, `print_dataclass`, `add_instrumentation_args`,
`make_cli_instrumentation`, `add_timings` -- the common `--progress`/
`--profile` argparse wiring and JSON-summary printing every subcommand uses.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from typing import Any

from touche.instrumentation import Instrumentation


def print_json(payload: Any) -> None:
    """Print `payload` as indented, sorted-key JSON -- the CLI's standard summary format."""
    print(json.dumps(payload, indent=2, sort_keys=True))


def print_dataclass(value: Any) -> None:
    """`print_json` a dataclass instance (e.g. `PairStats`) by converting it to a dict first."""
    if not is_dataclass(value):
        raise TypeError(f"Expected dataclass instance, got {type(value)!r}")
    print_json(asdict(value))


def add_instrumentation_args(parser: argparse.ArgumentParser) -> None:
    """Register the `--progress`/`--profile` flags shared by every subcommand that does real work."""
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
    """Build an `Instrumentation` from parsed `--progress`/`--profile` args."""
    return Instrumentation(
        progress=bool(getattr(args, "progress", False)),
        profile=bool(getattr(args, "profile", False)),
    )


def add_timings(payload: dict[str, Any], instrument: Instrumentation) -> dict[str, Any]:
    """Add a `"timings"` key to `payload` when `--profile` was passed; otherwise leave it unchanged."""
    if instrument.profile:
        payload["timings"] = instrument.timings
    return payload
