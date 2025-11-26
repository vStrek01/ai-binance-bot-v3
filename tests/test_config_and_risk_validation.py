import pytest

from bot.core import config
from bot.core.config import ConfigValidationError


def test_validate_configs_rejects_non_positive_per_trade_risk() -> None:
    config.risk.per_trade_risk = 0.0
    with pytest.raises(ConfigValidationError):
        config.validate_configs()


def test_validate_configs_rejects_negative_daily_loss_pct() -> None:
    config.risk.max_daily_loss_pct = -0.1
    with pytest.raises(ConfigValidationError):
        config.validate_configs()


def test_validate_configs_rejects_daily_loss_pct_above_one() -> None:
    config.risk.max_daily_loss_pct = 1.5
    with pytest.raises(ConfigValidationError):
        config.validate_configs()


def test_validate_configs_rejects_symbol_exposure_exceeding_account() -> None:
    config.risk.max_account_exposure = 0.2
    config.risk.max_symbol_exposure = 0.25
    with pytest.raises(ConfigValidationError):
        config.validate_configs()
