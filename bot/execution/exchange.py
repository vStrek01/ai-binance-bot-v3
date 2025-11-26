"""Exchange metadata utilities for Binance USDⓈ-M futures."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
from typing import Any, Dict, List, Optional

from bot.core.config import BotConfig, ensure_directories
from bot.execution.exchange_client import ExchangeClient
from bot.utils.fileio import atomic_write_text
from bot.utils.logger import get_logger

logger = get_logger(__name__)


def _round_down(value: float, step: float) -> float:
    if step == 0:
        return value
    quantized = Decimal(str(value)).quantize(Decimal(str(step)), rounding=ROUND_DOWN)
    return float(quantized)


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass(slots=True)
class SymbolFilters:
    min_qty: float
    min_notional: float
    step_size: float
    tick_size: float
    max_leverage: float
    quote_asset: str = "USDT"

    def adjust_quantity(self, qty: float) -> float:
        return _round_down(max(qty, 0.0), self.step_size)

    def adjust_price(self, price: float) -> float:
        return _round_down(max(price, 0.0), self.tick_size)


class ExchangeInfoManager:
    def __init__(
        self,
        cfg: BotConfig,
        client: Optional[ExchangeClient] = None,
        cache_file: Optional[Path] = None,
        prefetched: Optional[Dict[str, SymbolFilters]] = None,
    ) -> None:
        self._config = cfg
        self.client = client
        self.cache_path = cache_file or (cfg.paths.data_dir / "exchange_info.json")
        self.symbols: Dict[str, SymbolFilters] = prefetched or {}
        ensure_directories(cfg.paths, extra=[self.cache_path.parent])
        if not prefetched:
            self._load_cache()

    def _load_cache(self) -> None:
        if not self.cache_path.exists():
            return
        try:
            payload = json.loads(self.cache_path.read_text())
            for sym, raw in payload.items():
                try:
                    self.symbols[sym] = SymbolFilters(**raw)
                except TypeError:
                    logger.warning("Skipping corrupt symbol filter for %s", sym)
            logger.info("Loaded exchange info cache for %s symbols", len(self.symbols))
        except json.JSONDecodeError:
            logger.warning("Exchange info cache corrupted; refreshing metadata")

    def _save_cache(self) -> None:
        serialized = {sym: self._serialize_filters(filt) for sym, filt in self.symbols.items()}
        atomic_write_text(self.cache_path, json.dumps(serialized, indent=2))

    def refresh(self, force: bool = False) -> None:
        if self.symbols and not force:
            return
        if not self.client:
            raise RuntimeError("Exchange client unavailable for refresh")
        try:
            metadata = self.client.exchange_info()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Exchange info refresh failed: %s", exc)
            if not self.symbols:
                raise
            return
        symbols: List[Dict[str, Any]] = metadata.get("symbols", []) or []
        leverage_lookup = self._fetch_leverage_limits()
        updated: Dict[str, SymbolFilters] = {}
        allowed_quotes = {"USDC", "USDT"}
        for info in symbols:
            if info.get("contractType") != "PERPETUAL" or info.get("quoteAsset") not in allowed_quotes:
                continue
            raw_filters: List[Dict[str, Any]] = info.get("filters", []) or []
            filter_map = {flt["filterType"]: flt for flt in raw_filters if "filterType" in flt}
            lot = filter_map.get("LOT_SIZE", {})
            price = filter_map.get("PRICE_FILTER", {})
            notional = filter_map.get("MIN_NOTIONAL", {})
            symbol = str(info.get("symbol"))
            if not symbol:
                continue
            updated[symbol] = SymbolFilters(
                min_qty=_to_float(lot.get("minQty"), 0.0),
                min_notional=_to_float(notional.get("notional"), 0.0),
                step_size=_to_float(lot.get("stepSize"), 1.0),
                tick_size=_to_float(price.get("tickSize"), 0.01),
                max_leverage=leverage_lookup.get(symbol, 20.0),
                quote_asset=str(info.get("quoteAsset", "")),
            )
        self.symbols = updated
        self._save_cache()
        logger.info("Cached exchange info for %s USDC/USDT symbols", len(self.symbols))

    def _fetch_leverage_limits(self) -> Dict[str, float]:
        if not self.client or not hasattr(self.client, "leverage_bracket"):
            return {}
        try:
            brackets = self.client.leverage_bracket()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to pull leverage brackets: %s", exc)
            return {}
        lookup: Dict[str, float] = {}
        for entry in brackets or []:
            symbol = entry.get("symbol")
            bracket_info = (entry.get("brackets") or entry.get("bracket") or [])
            if not symbol or not bracket_info:
                continue
            first = bracket_info[0]
            leverage = _to_float(first.get("initialLeverage"), 20.0)
            lookup[symbol] = leverage
        return lookup

    def get_filters(self, symbol: str) -> Optional[SymbolFilters]:
        if symbol not in self.symbols and self.client:
            self.refresh(force=True)
        return self.symbols.get(symbol)

    def validate_order(self, symbol: str, price: float, quantity: float) -> tuple[bool, str, float]:
        filters = self.get_filters(symbol)
        if not filters:
            return False, "missing_filters", 0.0
        adjusted_qty = filters.adjust_quantity(quantity)
        adjusted_price = filters.adjust_price(price)
        notional = adjusted_qty * adjusted_price
        if adjusted_qty < filters.min_qty:
            return False, "min_qty", adjusted_qty
        if notional < filters.min_notional:
            return False, "min_notional", adjusted_qty
        return True, "ok", adjusted_qty

    def snapshot(self) -> Dict[str, Dict[str, float]]:
        return {sym: self._serialize_filters(filt) for sym, filt in self.symbols.items()}

    @staticmethod
    def _serialize_filters(filters: SymbolFilters) -> Dict[str, Any]:
        raw = asdict(filters)
        serialized: Dict[str, Any] = {}
        for key, value in raw.items():
            if isinstance(value, (int, float)):
                serialized[key] = float(value)
            else:
                serialized[key] = value
        return serialized


__all__ = ["ExchangeInfoManager", "SymbolFilters"]
