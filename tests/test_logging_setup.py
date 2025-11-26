import logging

from bot.core.config import load_config
from bot.utils.logger import RunContext, generate_run_id, get_logger, setup_logging


def test_setup_logging_creates_run_file(tmp_path):
    cfg = load_config(base_dir=tmp_path)
    run_id = generate_run_id("test")
    run_type = "backtest"
    setup_logging(cfg, run_context=RunContext(run_id=run_id, run_type=run_type))
    logger = get_logger(__name__)
    logger.info("log_file_check")
    log_file = tmp_path / "logs" / run_type / f"{run_id}.log"
    assert log_file.exists()
    contents = log_file.read_text()
    assert run_id in contents
    assert logging.getLogger().getEffectiveLevel() == logging.getLevelName(cfg.runtime.log_level)


def test_run_context_injected_into_logs(tmp_path, caplog):
    cfg = load_config(base_dir=tmp_path)
    run_context = RunContext(run_id="ctx-123", run_type="risk")
    setup_logging(cfg, run_context=run_context)
    logger = get_logger("bot.risk.engine")
    logger.info("risk_log", extra={"event": "sample"})
    log_file = cfg.paths.log_dir / run_context.run_type / f"{run_context.run_id}.log"
    contents = log_file.read_text()
    assert "ctx-123" in contents
    assert "risk_log" in contents
