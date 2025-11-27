"""Exchange metadata utilities for Binance USDⓈ-M futures."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
from typing import Any, Dict, List, Optional

from bot.core.config import BotConfig, ensure_directories
from bot.execution.exchange_client import ExchangeClient
from bot.utils.fileio import atomic_write_text
from bot.utils.logger import get_logger
from exchange.symbols import SymbolInfo, build_symbol_info, fetch_leverage_limits

logger = get_logger(__name__)


def _round_down(value: float, step: float) -> float:
    if step <= 0:
        return max(value, 0.0)
    quantized = Decimal(str(max(value, 0.0))).quantize(Decimal(str(step)), rounding=ROUND_DOWN)
    return float(quantized)


@dataclass(slots=True)
class SymbolFilters:
    min_qty: float
    min_notional: float
    step_size: float
    tick_size: float
    max_leverage: float
    quote_asset: str = "USDT"
    symbol: str = ""
    base_asset: str = ""
    _info: Optional[SymbolInfo] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self._info is None:
            symbol_code = (self.symbol or f"{self.base_asset}{self.quote_asset}").upper()
            base_asset = self.base_asset or self._infer_base(symbol_code, self.quote_asset)
            self._info = SymbolInfo(
                symbol=symbol_code,
                base_asset=base_asset,
                quote_asset=self.quote_asset,
                tick_size=Decimal(str(self.tick_size or 0)),
                step_size=Decimal(str(self.step_size or 0)),
                min_notional=Decimal(str(self.min_notional or 0)),
                min_qty=Decimal(str(self.min_qty or 0)),
                min_price=Decimal("0"),
                max_price=Decimal("0"),
                multiplier_up=Decimal("0"),
                multiplier_down=Decimal("0"),
                max_leverage=int(self.max_leverage or 0),
            )
        else:
            self.symbol = self._info.symbol
            self.base_asset = self._info.base_asset
            self.quote_asset = self._info.quote_asset
            self.min_qty = float(self._info.min_qty)
            self.min_notional = float(self._info.min_notional)
            self.step_size = float(self._info.step_size)
            self.tick_size = float(self._info.tick_size)
            self.max_leverage = float(self._info.max_leverage)

    def adjust_quantity(self, qty: float) -> float:
        if self._info:
            return self._info.round_qty(qty)
        return _round_down(max(qty, 0.0), self.step_size or 1.0)

    def adjust_price(self, price: float) -> float:
        if self._info:
            return self._info.round_price(price)
        return _round_down(max(price, 0.0), self.tick_size or 1.0)

    @property
    def symbol_info(self) -> SymbolInfo:
        if not self._info:
            raise RuntimeError("Symbol filters missing symbol_info binding")
        return self._info

    @classmethod
    def from_symbol_info(cls, info: SymbolInfo) -> SymbolFilters:
        return cls(
            min_qty=float(info.min_qty),
            min_notional=float(info.min_notional),
            step_size=float(info.step_size),
            tick_size=float(info.tick_size),
            max_leverage=float(info.max_leverage),
            quote_asset=info.quote_asset,
            symbol=info.symbol,
            base_asset=info.base_asset,
            _info=info,
        )

    @classmethod
    def synthetic(
        cls,
        *,
        symbol: str,
        base_asset: str,
        quote_asset: str,
        tick_size: float,
        step_size: float,
        min_qty: float,
        min_notional: float,
        max_leverage: float = 20.0,
    ) -> SymbolFilters:
        info = SymbolInfo(
            symbol=symbol,
            base_asset=base_asset,
            quote_asset=quote_asset,
            tick_size=Decimal(str(tick_size)),
            step_size=Decimal(str(step_size)),
            min_notional=Decimal(str(min_notional)),
            min_qty=Decimal(str(min_qty)),
            min_price=Decimal("0"),
            max_price=Decimal("0"),
            multiplier_up=Decimal("0"),
            multiplier_down=Decimal("0"),
            max_leverage=int(max_leverage),
        )
        return cls.from_symbol_info(info)

    @staticmethod
    def _infer_base(symbol: str, quote: str) -> str:
        if quote and symbol.upper().endswith(quote.upper()):
            candidate = symbol[: -len(quote)]
            return candidate or symbol
        return symbol


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
        self.symbols: Dict[str, SymbolFilters] = {}
        if prefetched:
            for sym, filt in prefetched.items():
                self.symbols[sym] = self._bind_symbol(sym, filt)
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
                    info_payload = raw.get("symbol_info") if isinstance(raw, dict) else None
                    if info_payload:
                        info = SymbolInfo.from_dict(info_payload)
                        self.symbols[sym] = SymbolFilters.from_symbol_info(info)
                    else:
                        logger.info("Discarding legacy exchange cache for %s; forcing refresh", sym)
                except (TypeError, ValueError):
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
        leverage_lookup = fetch_leverage_limits(self.client)
        updated: Dict[str, SymbolFilters] = {}
        allowed_quotes = {"USDC", "USDT"}
        for entry in symbols:
            if entry.get("quoteAsset") not in allowed_quotes:
                continue
            info = build_symbol_info(entry, leverage_lookup=leverage_lookup, require_contract="PERPETUAL")
            if not info:
                continue
            updated[info.symbol] = SymbolFilters.from_symbol_info(info)
        self.symbols = updated
        self._save_cache()
        logger.info("Cached exchange info for %s USDC/USDT symbols", len(self.symbols))

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

    def snapshot(self) -> Dict[str, Dict[str, Any]]:
        return {sym: self._serialize_filters(filt) for sym, filt in self.symbols.items()}

    @staticmethod
    def _serialize_filters(filters: SymbolFilters) -> Dict[str, Any]:
        info = filters.symbol_info
        return {
            "min_qty": float(filters.min_qty),
            "min_notional": float(filters.min_notional),
            "step_size": float(filters.step_size),
            "tick_size": float(filters.tick_size),
            "max_leverage": float(filters.max_leverage),
            "quote_asset": filters.quote_asset,
            "symbol": filters.symbol,
            "base_asset": filters.base_asset,
            "symbol_info": info.to_dict(),
        }

    @staticmethod
    def _bind_symbol(symbol: str, filters: SymbolFilters) -> SymbolFilters:
        info = filters.symbol_info
        if info.symbol == symbol:
            return filters
        base_asset = info.base_asset or SymbolFilters._infer_base(symbol, info.quote_asset)
        rebound = SymbolInfo(
            symbol=symbol,
            base_asset=base_asset,
            quote_asset=info.quote_asset,
            tick_size=info.tick_size,
            step_size=info.step_size,
            min_notional=info.min_notional,
            min_qty=info.min_qty,
            min_price=info.min_price,
            max_price=info.max_price,
            multiplier_up=info.multiplier_up,
            multiplier_down=info.multiplier_down,
            max_leverage=info.max_leverage,
        )
        return SymbolFilters.from_symbol_info(rebound)


__all__ = ["ExchangeInfoManager", "SymbolFilters"]
