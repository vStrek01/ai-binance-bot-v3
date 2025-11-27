from __future__ import annotations

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


class SymbolResolver:
    def __init__(self, client: Any, *, symbols: Optional[Iterable[str]] = None) -> None:
        self._client = client
        self._symbols: Dict[str, SymbolInfo] = {}
        self._allowlist = {sym.upper() for sym in symbols} if symbols else None

    def refresh(self) -> None:
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
            tick_size = Decimal(str(price.get("tickSize", "0"))) or Decimal("0")
            step_size = Decimal(str(lot.get("stepSize", "0"))) or Decimal("0")
            min_qty = Decimal(str(lot.get("minQty", "0"))) or Decimal("0")
            min_notional = Decimal(str(notional.get("notional", "0"))) or Decimal("0")
            refreshed[symbol_code] = SymbolInfo(
                symbol=symbol_code,
                base_asset=str(entry.get("baseAsset", "")),
                quote_asset=str(entry.get("quoteAsset", "")),
                tick_size=tick_size,
                step_size=step_size,
                min_notional=min_notional,
                min_qty=min_qty,
            )
        if self._allowlist:
            missing = self._allowlist.difference(refreshed)
            if missing:
                raise ValueError(f"Missing exchange symbols: {sorted(missing)}")
        self._symbols = refreshed

    def get(self, symbol: str) -> SymbolInfo:
        key = symbol.upper()
        try:
            return self._symbols[key]
        except KeyError as exc:
            raise ValueError(f"Unknown symbol: {symbol}") from exc

    def snapshot(self) -> Dict[str, SymbolInfo]:
        return dict(self._symbols)
