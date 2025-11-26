from core.safety import SafetyLimits, SafetyState, cap_notional, check_kill_switch


def _limits() -> SafetyLimits:
    return SafetyLimits(max_daily_drawdown_pct=5.0, max_total_notional_usd=20_000.0, max_consecutive_losses=3)


def test_kill_switch_triggers_on_drawdown():
    limits = _limits()
    state = SafetyState(daily_start_equity=10_000.0, current_equity=9_400.0, consecutive_losses=0)
    assert check_kill_switch(limits, state)


def test_kill_switch_triggers_on_consecutive_losses():
    limits = _limits()
    state = SafetyState(daily_start_equity=10_000.0, current_equity=9_800.0, consecutive_losses=3)
    assert check_kill_switch(limits, state)


def test_notional_cap_respects_symbol_limit():
    qty = cap_notional(
        requested_qty=5,
        price=1_000.0,
        max_symbol_notional=4_000.0,
        max_total_notional=20_000.0,
        current_symbol_notional=3_500.0,
        current_total_notional=3_500.0,
    )
    assert round(qty * 1_000.0, 2) == 500.0


def test_notional_cap_respects_global_limit():
    qty = cap_notional(
        requested_qty=20,
        price=1_000.0,
        max_symbol_notional=50_000.0,
        max_total_notional=25_000.0,
        current_symbol_notional=0.0,
        current_total_notional=24_000.0,
    )
    assert round(qty * 1_000.0, 2) == 1_000.0
