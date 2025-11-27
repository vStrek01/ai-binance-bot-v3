from __future__ import annotations

import time
from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN
from typing import Any, Dict, Iterable, Optional


@dataclass(slots=True)
class SymbolInfo:
    symbol: str
    base_asset: str
    quote_asset: str
    tick_size: Decimal
    step_size: Decimal
    min_notional: Decimal
    min_qty: Decimal
    min_price: Decimal = Decimal("0")
    max_price: Decimal = Decimal("0")
    multiplier_up: Decimal = Decimal("0")
    multiplier_down: Decimal = Decimal("0")
    max_leverage: int = 20

    def round_price(self, price: float) -> float:
        if price <= 0:
            return price
        d_price = Decimal(str(price))
        if self.tick_size <= 0:
            return float(d_price)
        normalized = (d_price / self.tick_size).to_integral_value(rounding=ROUND_DOWN) * self.tick_size
        return float(normalized)

    def round_qty(self, qty: float) -> float:
        if qty <= 0:
            return 0.0
        d_qty = Decimal(str(qty))
        if self.step_size <= 0:
            return float(d_qty)
        normalized = (d_qty / self.step_size).to_integral_value(rounding=ROUND_DOWN) * self.step_size
        return float(normalized)

    def validate_notional(self, price: float, qty: float) -> bool:
        if price <= 0 or qty <= 0:
            return False
        notional = Decimal(str(price)) * Decimal(str(qty))
        return notional >= self.min_notional

    def clamp_price(self, price: float) -> float:
        rounded = self.round_price(price)
        if self.min_price > 0 and rounded < float(self.min_price):
            rounded = float(self.min_price)
        if self.max_price > 0 and rounded > float(self.max_price):
            rounded = float(self.max_price)
        return rounded

    def clamp_qty(self, qty: float) -> float:
        rounded = self.round_qty(qty)
        if self.min_qty > 0 and rounded < float(self.min_qty):
            return 0.0
        return rounded


class SymbolResolver:
    def __init__(
        self,
        client: Any,
        *,
        symbols: Optional[Iterable[str]] = None,
        ttl_seconds: int = 900,
    ) -> None:
        self._client = client
        self._symbols: Dict[str, SymbolInfo] = {}
        self._allowlist = {sym.upper() for sym in symbols} if symbols else None
        self._ttl = max(ttl_seconds, 0)
        self._last_refresh: Optional[float] = None

    def refresh(self, *, force: bool = False) -> None:
        if not force and self._last_refresh and self._ttl > 0:
            if (time.time() - self._last_refresh) < self._ttl and self._symbols:
                return
        payload = self._client.get_exchange_info()
        symbols = payload.get("symbols", []) if isinstance(payload, dict) else []
        refreshed: Dict[str, SymbolInfo] = {}
        for entry in symbols:
            symbol = (entry or {}).get("symbol")
            if not symbol:
                continue
            symbol_code = str(symbol).upper()
            if self._allowlist and symbol_code not in self._allowlist:
                continue
            if entry.get("contractType") != "PERPETUAL":
                continue
            filters = {flt.get("filterType"): flt for flt in (entry.get("filters") or [])}
            lot = filters.get("LOT_SIZE", {})
            price = filters.get("PRICE_FILTER", {})
            notional = filters.get("MIN_NOTIONAL", {})
            percent = filters.get("PERCENT_PRICE", {})
            tick_size = Decimal(str(price.get("tickSize", "0"))) or Decimal("0")
            step_size = Decimal(str(lot.get("stepSize", "0"))) or Decimal("0")
            min_qty = Decimal(str(lot.get("minQty", "0"))) or Decimal("0")
            min_notional = Decimal(str(notional.get("notional", "0"))) or Decimal("0")
            min_price = Decimal(str(price.get("minPrice", "0"))) or Decimal("0")
            max_price = Decimal(str(price.get("maxPrice", "0"))) or Decimal("0")
            multiplier_up = Decimal(str(percent.get("multiplierUp", "0"))) or Decimal("0")
            multiplier_down = Decimal(str(percent.get("multiplierDown", "0"))) or Decimal("0")
            max_leverage = int(entry.get("defaultLeverage", 20) or 20)
            refreshed[symbol_code] = SymbolInfo(
                symbol=symbol_code,
                base_asset=str(entry.get("baseAsset", "")),
                quote_asset=str(entry.get("quoteAsset", "")),
                tick_size=tick_size,
                step_size=step_size,
                min_notional=min_notional,
                min_qty=min_qty,
                min_price=min_price,
                max_price=max_price,
                multiplier_up=multiplier_up,
                multiplier_down=multiplier_down,
                max_leverage=max_leverage,
            )
        if self._allowlist:
            missing = self._allowlist.difference(refreshed)
            if missing:
                raise ValueError(f"Missing exchange symbols: {sorted(missing)}")
        self._symbols = refreshed
        self._last_refresh = time.time()

    def get(self, symbol: str) -> SymbolInfo:
        key = symbol.upper()
        try:
            return self._symbols[key]
        except KeyError as exc:
            raise ValueError(f"Unknown symbol: {symbol}") from exc

    def snapshot(self) -> Dict[str, SymbolInfo]:
        return dict(self._symbols)

    def round_price(self, symbol: str, price: float) -> float:
        return self.get(symbol).round_price(price)

    def round_quantity(self, symbol: str, qty: float) -> float:
        return self.get(symbol).round_qty(qty)

    def describe(self, symbol: str) -> Dict[str, Any]:
        info = self.get(symbol)
        return {
            "symbol": info.symbol,
            "tick_size": float(info.tick_size),
            "step_size": float(info.step_size),
            "min_notional": float(info.min_notional),
            "min_qty": float(info.min_qty),
            "max_leverage": info.max_leverage,
        }
