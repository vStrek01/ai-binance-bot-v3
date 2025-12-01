from __future__ import annotations

from types import SimpleNamespace

import bot.runner as runner_mod
from bot.core.config import load_config


def _clear_scalping_env(monkeypatch):
    monkeypatch.delenv("SCALPING_SYMBOLS", raising=False)
    monkeypatch.delenv("SCALPING_SYMBOL", raising=False)


def test_scalping_symbols_prefer_env_csv(monkeypatch, tmp_path):
    cfg = load_config(base_dir=tmp_path)
    _clear_scalping_env(monkeypatch)
    monkeypatch.setenv("SCALPING_SYMBOLS", " BTCUSDT , ETHUSDT , , LTCUSDT ")

    symbols = runner_mod._scalping_symbol_defaults(cfg)

    assert symbols == ["BTCUSDT", "ETHUSDT", "LTCUSDT"]


def test_scalping_symbols_support_single_symbol_env(monkeypatch, tmp_path):
    cfg = load_config(base_dir=tmp_path)
    _clear_scalping_env(monkeypatch)
    monkeypatch.setenv("SCALPING_SYMBOL", "xrpusdt")

    symbols = runner_mod._scalping_symbol_defaults(cfg)

    assert symbols == ["XRPUSDT"]


def test_scalping_symbols_fall_back_to_config(monkeypatch, tmp_path):
    cfg = load_config(base_dir=tmp_path)
    cfg = cfg.model_copy(update={"symbols": ["adausdt", "ethusdt", "adausdt"]})
    _clear_scalping_env(monkeypatch)

    symbols = runner_mod._scalping_symbol_defaults(cfg)

    assert symbols == ["ADAUSDT", "ETHUSDT"]


def test_scalping_symbols_default_to_constant(monkeypatch, tmp_path):
    cfg = load_config(base_dir=tmp_path)
    cfg = cfg.model_copy(
        update=
        {
            "symbols": [],
            "universe": cfg.universe.model_copy(update={"demo_symbols": [], "default_symbols": []}),
        }
    )
    _clear_scalping_env(monkeypatch)

    symbols = runner_mod._scalping_symbol_defaults(cfg)

    assert symbols == list(runner_mod._DEFAULT_SCALPING_SYMBOLS)


def test_resolve_symbols_uses_fallback_when_args_missing(monkeypatch, tmp_path):
    cfg = load_config(base_dir=tmp_path)
    _clear_scalping_env(monkeypatch)
    fallback = ["ETHUSDT", "BTCUSDT"]
    exchange = SimpleNamespace(symbols={sym: {} for sym in fallback})

    resolved = runner_mod._resolve_symbols(
        cfg,
        symbol=None,
        symbols_arg=None,
        exchange=exchange,
        demo_mode=True,
        fallback_symbols=fallback,
    )

    assert resolved == fallback
