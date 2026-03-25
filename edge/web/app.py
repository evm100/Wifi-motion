"""
app.py -- FastAPI web dashboard for ESP32 CSI node monitoring & data collection.

Run:
    python -m edge.web.app [--port 8080] [--udp-port 5005]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Ensure repo root is importable.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from proto.constants import UDP_PORT
from edge.web.monitor import NodeMonitor
from edge.web.collector import Collector

logger = logging.getLogger("edge.web")

app = FastAPI(title="CSI Dashboard")

# ── Shared state (created at startup) ────────────────────────────────
monitor: NodeMonitor | None = None
collector: Collector | None = None

STATIC_DIR = Path(__file__).parent / "static"


# ── Pydantic request models ──────────────────────────────────────────
class StartSessionRequest(BaseModel):
    activities: list[str] | None = None
    duration: float = 30.0
    repetitions: int = 1
    countdown: float = 5.0


# ── REST endpoints ───────────────────────────────────────────────────
@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/nodes")
async def get_nodes():
    return monitor.get_status()


@app.get("/api/collect/status")
async def collect_status():
    return collector.get_status()


@app.post("/api/collect/start")
async def collect_start(req: StartSessionRequest):
    return collector.start_session(
        activities=req.activities,
        duration=req.duration,
        repetitions=req.repetitions,
        countdown=req.countdown,
    )


@app.post("/api/collect/stop")
async def collect_stop():
    return collector.stop_session()


@app.post("/api/collect/skip")
async def collect_skip():
    return collector.skip_current()


@app.get("/api/sessions")
async def list_sessions():
    return collector.list_sessions()


# ── WebSocket for live updates ────────────────────────────────────────
@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    q = monitor.subscribe()
    try:
        while True:
            node_status = await q.get()
            collect_status = collector.get_status()
            await websocket.send_json({
                "nodes": node_status,
                "collection": collect_status,
            })
    except WebSocketDisconnect:
        pass
    finally:
        monitor.unsubscribe(q)


# ── Lifecycle ─────────────────────────────────────────────────────────
_tick_task: asyncio.Task | None = None


async def _collector_tick_loop():
    """Drive the collector state machine and feed packets to it."""
    while True:
        await asyncio.sleep(0.5)
        collector.tick()


@app.on_event("startup")
async def startup():
    global monitor, collector, _tick_task

    udp_port = int(app.state.udp_port) if hasattr(app.state, "udp_port") else UDP_PORT
    data_dir = app.state.data_dir if hasattr(app.state, "data_dir") else "data/sessions"

    monitor = NodeMonitor()
    collector = Collector(output_dir=data_dir)

    # Hook collector into monitor so recording packets get written to disk.
    _original_handle = monitor.handle_packet

    def _combined_handle(data: bytes) -> None:
        _original_handle(data)
        collector.handle_packet(data)

    monitor.handle_packet = _combined_handle

    await monitor.start(port=udp_port)
    _tick_task = asyncio.create_task(_collector_tick_loop())
    logger.info("Dashboard started — UDP :%d, data → %s", udp_port, data_dir)


@app.on_event("shutdown")
async def shutdown():
    if _tick_task:
        _tick_task.cancel()
    if monitor:
        await monitor.stop()


# ── Static files (mount after routes so /api etc. take precedence) ───
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── CLI entry point ──────────────────────────────────────────────────
def main():
    import uvicorn

    parser = argparse.ArgumentParser(description="CSI Dashboard")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--udp-port", type=int, default=UDP_PORT)
    parser.add_argument("--data-dir", default="data/sessions")
    args = parser.parse_args()

    app.state.udp_port = args.udp_port
    app.state.data_dir = args.data_dir

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
