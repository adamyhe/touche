"""Top-level `touche` CLI entry point: wires each command group's parser together."""

from __future__ import annotations

import argparse

from touche import __version__
from touche.cli.apa import add_apa_parser
from touche.cli.background import add_background_parser
from touche.cli.local_decay import add_local_decay_parser
from touche.cli.preprocess import add_preprocess_parser
from touche.cli.utils import ToucheArgumentParser


def build_parser() -> argparse.ArgumentParser:
    """Assemble the `touche` argparse parser with every command group's subcommands attached."""
    parser = ToucheArgumentParser(
        prog="touche",
        description=(
            "Analyze enhancer-promoter contacts from processed pairs files. "
            "Use '<command> --help' for workflow-specific options."
        ),
    )
    parser.add_argument("--version", action="version", version=f"touche {__version__}")

    subparsers = parser.add_subparsers(dest="command", required=True)
    add_preprocess_parser(subparsers)
    add_local_decay_parser(subparsers)
    add_background_parser(subparsers)
    add_apa_parser(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse `argv` and dispatch to the matched subcommand's `func` callback."""
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0
