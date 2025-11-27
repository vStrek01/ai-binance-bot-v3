"""Quick smoke test for loading the demo-live configuration."""
from pathlib import Path
from sys import exit

from infra.config_loader import ConfigError, load_app_config


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    try:
        config = load_app_config(
            path="config.yaml",
            base_dir=base_dir,
            mode_override="demo-live",
        )
    except ConfigError as exc:
        raise SystemExit(f"CONFIG_FAILED: {exc}") from exc

    if not config.exchange.use_testnet:
        raise SystemExit("CONFIG_FAILED: expected use_testnet=true in demo-live mode")

    print("CONFIG_OK")


if __name__ == "__main__":
    main()
