from __future__ import annotations

import argparse

import pytest

from bot.runner import build_parser


def test_build_parser_succeeds() -> None:
    parser = build_parser()
    assert isinstance(parser, argparse.ArgumentParser)


@pytest.mark.parametrize(
    "argv",
    [
        ["download", "--symbols", "BTCUSDT", "--intervals", "1m"],
        ["backtest", "--symbol", "BTCUSDT", "--interval", "1m"],
        ["optimize"],
        ["train-all"],
        ["self-tune", "--symbols", "BTCUSDT", "--intervals", "1m"],
        ["dry-run", "--symbol", "BTCUSDT", "--interval", "1m"],
        ["dry-run-portfolio", "--interval", "1m"],
        ["demo-live", "--symbol", "BTCUSDT", "--interval", "1m"],
        ["train-rl", "--symbol", "BTCUSDT", "--interval", "1m"],
        ["api", "--host", "127.0.0.1", "--port", "8000"],
        ["full-cycle"],
    ],
)
def test_subcommands_define_callable(argv: list[str]) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    assert hasattr(args, "func")