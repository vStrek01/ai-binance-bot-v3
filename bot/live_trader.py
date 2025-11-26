"""Compatibility shim exposing the LiveTrader from bot.execution.live."""
from bot.execution.live import LiveTrader  # noqa: F401

__all__ = ["LiveTrader"]
