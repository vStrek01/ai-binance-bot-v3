"""Standalone diagnostic for Binance Futures demo account access."""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import requests
from dotenv import load_dotenv

BASE_URL = "https://demo-fapi.binance.com"
ACCOUNT_PATH = "/fapi/v2/account"


def _load_local_env() -> None:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=False)


_load_local_env()


def _env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        print(f"Missing required environment variable: {name}")
        sys.exit(1)
    return value

def _sign(secret: str, query: str) -> str:
    signature = hmac.new(secret.encode(), query.encode(), hashlib.sha256)
    return signature.hexdigest()

def _asset_keys(payload: Dict[str, Any]) -> Iterable[str]:
    records = payload.get("assets") or payload.get("balances") or []
    typed_records: List[Dict[str, Any]] = [entry for entry in records if isinstance(entry, dict)]
    return [str(entry.get("asset", "")) for entry in typed_records]

def call_account_endpoint() -> int:
    api_key = _env("BINANCE_API_KEY")
    api_secret = _env("BINANCE_API_SECRET")

    params = {
        "timestamp": int(time.time() * 1000),
        "recvWindow": 5000,
    }
    query = "&".join(f"{key}={value}" for key, value in params.items())
    signature = _sign(api_secret, query)
    signed_query = f"{query}&signature={signature}"
    url = f"{BASE_URL}{ACCOUNT_PATH}?{signed_query}"

    headers = {
        "X-MBX-APIKEY": api_key,
        "User-Agent": "demo-account-diagnostic",
    }

    print(f"Request URL: {url}")
    response = requests.get(url, headers=headers, timeout=10)
    print(f"Status: {response.status_code}")
    print("Response body:")
    print(response.text)

    if response.status_code == 200:
        try:
            payload: Dict[str, Any] = response.json()
        except json.JSONDecodeError:
            print("Warning: Unable to decode JSON payload")
        else:
            assets = list(_asset_keys(payload))
            print("Asset keys:")
            print(json.dumps(assets, indent=2))
    else:
        print("Error body echoed above; investigate credentials or permissions.")

    return response.status_code

def main() -> None:
    status = call_account_endpoint()
    sys.exit(0 if status == 200 else 1)

if __name__ == "__main__":
    main()
