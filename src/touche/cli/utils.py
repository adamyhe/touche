"""Shared helpers used by every `touche.cli.*` command module.

Public API (all of it, since this module is itself CLI-internal plumbing):
`ToucheArgumentParser`, `print_json`, `add_instrumentation_args`,
`make_cli_instrumentation`, `add_timings` -- the common `--progress`/
`--profile` argparse wiring and JSON-summary printing every subcommand uses.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from touche.instrumentation import Instrumentation


class ToucheHelpFormatter(argparse.HelpFormatter):
    """CLI help formatter that shows useful defaults without clutter."""

    def _get_help_string(self, action: argparse.Action) -> str:
        help_text = action.help or ""
        if "%(default)" in help_text:
            return help_text
        if action.required or action.default in (None, False, argparse.SUPPRESS):
            return help_text
        return f"{help_text} (default: {format_default(action.default)})"


class ToucheArgumentParser(argparse.ArgumentParser):
    """ArgumentParser with the project help formatter applied by default."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("formatter_class", ToucheHelpFormatter)
        super().__init__(*args, **kwargs)


def format_default(value: Any) -> str:
    """Render argparse defaults the way they should appear in CLI help."""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def print_json(payload: Any) -> None:
    """Print `payload` as indented, sorted-key JSON -- the CLI's standard summary format."""
    print(json.dumps(payload, indent=2, sort_keys=True))


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


def require_anchors_or_pair_list(args: argparse.Namespace) -> None:
    """Fail fast when a command needing a pair universe was given neither source.

    `--baits`/`--preys` and `--pairs-list` are alternative ways to say what
    pairs to analyze, so neither can be `required=True` on its own; argparse
    would otherwise let a command run with no pair universe at all and
    produce a silently empty result.
    """
    if getattr(args, "pairs_list", None) is not None:
        return
    if args.baits is None or args.preys is None:
        raise SystemExit("error: provide either --baits and --preys, or --pairs-list")
