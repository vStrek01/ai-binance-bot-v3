"""FastAPI application exposing the bot status and dashboard assets."""
from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import Response
from websockets.exceptions import ConnectionClosedOK

from bot.status import status_store
from bot.utils.logger import get_logger

logger = get_logger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIR = BASE_DIR / "frontend"
FRONTEND_DIR.mkdir(parents=True, exist_ok=True)
INDEX_FILE = FRONTEND_DIR / "index.html"


class NoCacheStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope: dict[str, Any]) -> Response:  # type: ignore[override]
        response = await super().get_response(path, scope)
        self._apply_no_cache_headers(response)
        return response

    @staticmethod
    def _apply_no_cache_headers(response: Response) -> None:
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"

app = FastAPI(title="AI Binance Bot v3 Dashboard", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", NoCacheStaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
async def dashboard_root() -> HTMLResponse:
    if INDEX_FILE.exists():
        response = HTMLResponse(INDEX_FILE.read_text(encoding="utf-8"))
    else:
        response = HTMLResponse("<h1>Dashboard assets not found</h1>", status_code=200)
    NoCacheStaticFiles._apply_no_cache_headers(response)
    return response


@app.get("/api/status")
async def api_status() -> Dict[str, Any]:
    return status_store.snapshot()


@app.websocket("/ws/status")
async def websocket_status(websocket: WebSocket) -> None:
    await websocket.accept()
    logger.info("Status websocket connected from %s", websocket.client)
    try:
        while True:
            await websocket.send_json(status_store.snapshot())
            await asyncio.sleep(1.0)
    except (WebSocketDisconnect, ConnectionClosedOK):
        logger.info("Status websocket disconnected cleanly")
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Status websocket terminated unexpectedly: %s", exc)
    finally:
        with contextlib.suppress(Exception):
            await websocket.close()


__all__ = ["app"]
