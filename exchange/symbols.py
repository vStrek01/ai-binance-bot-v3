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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "base_asset": self.base_asset,
            "quote_asset": self.quote_asset,
            "tick_size": str(self.tick_size),
            "step_size": str(self.step_size),
            "min_notional": str(self.min_notional),
            "min_qty": str(self.min_qty),
            "min_price": str(self.min_price),
            "max_price": str(self.max_price),
            "multiplier_up": str(self.multiplier_up),
            "multiplier_down": str(self.multiplier_down),
            "max_leverage": self.max_leverage,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> SymbolInfo:
        return cls(
            symbol=str(payload.get("symbol", "")),
            base_asset=str(payload.get("base_asset", "")),
            quote_asset=str(payload.get("quote_asset", "")),
            tick_size=Decimal(str(payload.get("tick_size", "0")) or "0"),
            step_size=Decimal(str(payload.get("step_size", "0")) or "0"),
            min_notional=Decimal(str(payload.get("min_notional", "0")) or "0"),
            min_qty=Decimal(str(payload.get("min_qty", "0")) or "0"),
            min_price=Decimal(str(payload.get("min_price", "0")) or "0"),
            max_price=Decimal(str(payload.get("max_price", "0")) or "0"),
            multiplier_up=Decimal(str(payload.get("multiplier_up", "0")) or "0"),
            multiplier_down=Decimal(str(payload.get("multiplier_down", "0")) or "0"),
            max_leverage=int(payload.get("max_leverage", 20) or 20),
        )

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


def _decimal_from(value: Any) -> Decimal:
    try:
        return Decimal(str(value))
    except (ArithmeticError, ValueError):
        return Decimal("0")


def _float_from(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def build_symbol_info(
    entry: Dict[str, Any],
    *,
    leverage_lookup: Optional[Dict[str, float]] = None,
    require_contract: Optional[str] = "PERPETUAL",
) -> Optional[SymbolInfo]:
    symbol = (entry or {}).get("symbol")
    if not symbol:
        return None
    symbol_code = str(symbol).upper()
    if require_contract and entry.get("contractType") != require_contract:
        return None
    filters = {flt.get("filterType"): flt for flt in (entry.get("filters") or [])}
    lot = filters.get("LOT_SIZE", {})
    price = filters.get("PRICE_FILTER", {})
    notional = filters.get("MIN_NOTIONAL", {})
    percent = filters.get("PERCENT_PRICE", {})
    leverage = None
    if leverage_lookup:
        leverage = leverage_lookup.get(symbol_code)
    if leverage is None:
        leverage = entry.get("defaultLeverage", 20)
    return SymbolInfo(
        symbol=symbol_code,
        base_asset=str(entry.get("baseAsset", "")),
        quote_asset=str(entry.get("quoteAsset", "")),
        tick_size=_decimal_from(price.get("tickSize")),
        step_size=_decimal_from(lot.get("stepSize")),
        min_notional=_decimal_from(notional.get("notional")),
        min_qty=_decimal_from(lot.get("minQty")),
        min_price=_decimal_from(price.get("minPrice")),
        max_price=_decimal_from(price.get("maxPrice")),
        multiplier_up=_decimal_from(percent.get("multiplierUp")),
        multiplier_down=_decimal_from(percent.get("multiplierDown")),
        max_leverage=int(leverage or 20),
    )


def fetch_leverage_limits(client: Any) -> Dict[str, float]:
    if not client or not hasattr(client, "leverage_bracket"):
        return {}
    try:
        brackets = client.leverage_bracket()
    except Exception:  # noqa: BLE001 - surfaced by callers via empty map
        return {}
    lookup: Dict[str, float] = {}
    for entry in brackets or []:
        symbol = entry.get("symbol")
        bracket_info = entry.get("brackets") or entry.get("bracket") or []
        if not symbol or not bracket_info:
            continue
        first = bracket_info[0]
        leverage = _float_from(first.get("initialLeverage"), 20.0)
        lookup[str(symbol).upper()] = leverage
    return lookup


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
        leverage_lookup = fetch_leverage_limits(self._client)
        refreshed: Dict[str, SymbolInfo] = {}
        for entry in symbols:
            info = build_symbol_info(entry, leverage_lookup=leverage_lookup)
            if not info:
                continue
            if self._allowlist and info.symbol not in self._allowlist:
                continue
            refreshed[info.symbol] = info
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
