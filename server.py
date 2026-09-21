"""FastAPI Server-Sent Events (SSE) & Telemetry Dispatcher for Project Outpost.

Provides:
- GET /events: Real-time SSE stream for web UI clients
- GET /events/recent: Ring-buffer snapshot for immediate client state
- POST /api/events: Ingest endpoint for wildlife_cv_runner (4090 / CUDA)
- GET /api/streams: Active streams registry, status, and telemetry
- GET /api/stats/species: Aggregated species sightings and frequency breakdown
- GET /api/alerts: Target species alert configuration rules
- POST /api/alerts: Register new target species alert rule
- GET /api/alerts/recent: Ring buffer of triggered alerts
- GET /health: Service telemetry & connection health
- Static snapshot file serving under /snapshots/
"""

import asyncio
import json
import os
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from alerts_dispatcher import AlertDispatcher
from retention import get_snapshots_storage_stats, prune_snapshots
from stream_watchdog import StreamWatchdog
from supervisor import get_hardware_diagnostics

BASE_DIR = Path(__file__).resolve().parent
SNAPSHOTS_DIR = BASE_DIR / "snapshots"
SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
INDEX_HTML = BASE_DIR / "index.html"

app = FastAPI(
    title="Outpost Wildlife CV Telemetry Bus",
    description="Real-time SSE event dispatcher and ring buffer for public wildlife livestreams",
    version="0.2.0",
)

# Enable CORS for local dev and staged frontends (Vite, Next.js, GitHub Pages)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static file mount for snapshot JPEG frames
app.mount("/snapshots", StaticFiles(directory=str(SNAPSHOTS_DIR)), name="snapshots")


@app.get("/", response_class=FileResponse)
async def root_dashboard():
    """Serves the single-page Ops Room surveillance frontend."""
    if INDEX_HTML.exists():
        return FileResponse(str(INDEX_HTML))
    return {"message": "Project Outpost Telemetry Bus Online"}


class DetectionEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: f"evt_{uuid.uuid4().hex[:8]}")
    timestamp: str = Field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    stream_id: str
    species: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: List[int] = Field(description="[x1, y1, x2, y2] bounding box coordinates")
    snapshot_url: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AlertRule(BaseModel):
    rule_id: str = Field(default_factory=lambda: f"rule_{uuid.uuid4().hex[:6]}")
    species: str
    min_confidence: float = 0.75
    stream_id: Optional[str] = None
    enabled: bool = True
    created_at: str = Field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


# In-memory stream registry
KNOWN_STREAMS = {
    "anacapa_kelp_01": {"name": "Anacapa Island Kelp Forest", "provider": "Explore.org", "fps_target": 120},
    "cornell_feeder_01": {"name": "Cornell Lab FeederWatch", "provider": "Cornell Lab", "fps_target": 60},
    "katmai_brooks_01": {"name": "Katmai Brooks Falls", "provider": "Explore.org", "fps_target": 60},
}

stream_telemetry: Dict[str, Dict[str, Any]] = {}
for sid, info in KNOWN_STREAMS.items():
    stream_telemetry[sid] = {
        **info,
        "stream_id": sid,
        "status": "idle",
        "total_detections": 0,
        "last_detection": None,
    }

species_stats: Dict[str, Dict[str, Any]] = {}
alert_rules: List[AlertRule] = [
    AlertRule(species="brown_bear", min_confidence=0.80),
    AlertRule(species="bald_eagle", min_confidence=0.75),
]
recent_alerts: deque = deque(maxlen=50)
alert_dispatcher = AlertDispatcher(default_cooldown_seconds=300.0)
stream_watchdog = StreamWatchdog()
stream_watchdog.register_stream("anacapa_kelp_01", "https://www.youtube.com/watch?v=OAJF1Ie1m_Q", "Explore.org")
stream_watchdog.register_stream("cornell_feeder_01", "https://www.youtube.com/watch?v=x10vL6_47Dw", "Cornell Lab")
stream_watchdog.register_stream("katmai_brooks_01", "https://www.youtube.com/watch?v=J7ZrIDvqlic", "Explore.org")

# In-memory ring buffer of recent events (depth: 200)
recent_events: deque = deque(maxlen=200)
subscribers: Set[asyncio.Queue] = set()
total_events_dispatched = 0
start_time = time.time()


@app.get("/health")
async def health_check():
    """System telemetry and active subscriber metrics."""
    snap_stats = get_snapshots_storage_stats(SNAPSHOTS_DIR)
    hw = get_hardware_diagnostics()
    return {
        "status": "online",
        "service": "outpost-telemetry-bus",
        "uptime_seconds": round(time.time() - start_time, 2),
        "active_sse_subscribers": len(subscribers),
        "ring_buffer_depth": len(recent_events),
        "total_events_dispatched": total_events_dispatched,
        "active_streams": len([s for s in stream_telemetry.values() if s["status"] == "active"]),
        "snapshot_storage_mb": snap_stats["total_mb"],
        "snapshot_files_count": snap_stats["total_files"],
        "hardware": hw,
    }


@app.get("/api/supervisor")
async def get_supervisor_status():
    """Returns supervisor hardware capability diagnostics and recommended execution mode."""
    return get_hardware_diagnostics()



@app.post("/api/maintenance/prune")
async def trigger_prune(max_age_hours: float = 24.0, max_storage_mb: float = 500.0):
    """Manually triggers snapshot retention pruning."""
    res = prune_snapshots(SNAPSHOTS_DIR, max_age_hours=max_age_hours, max_storage_mb=max_storage_mb)
    return {"status": "success", **res}


@app.get("/metrics", response_class=Response)
async def prometheus_metrics():
    """Exposes Prometheus text exposition format metrics for scraping."""
    snap_stats = get_snapshots_storage_stats(SNAPSHOTS_DIR)
    uptime = round(time.time() - start_time, 2)
    storage_bytes = int(snap_stats["total_mb"] * 1024 * 1024)

    lines = [
        "# HELP outpost_uptime_seconds Total runtime of Outpost telemetry server in seconds",
        "# TYPE outpost_uptime_seconds gauge",
        f"outpost_uptime_seconds {uptime}",
        "# HELP outpost_events_dispatched_total Cumulative detection events dispatched",
        "# TYPE outpost_events_dispatched_total counter",
        f"outpost_events_dispatched_total {total_events_dispatched}",
        "# HELP outpost_sse_subscribers_active Current active SSE client connections",
        "# TYPE outpost_sse_subscribers_active gauge",
        f"outpost_sse_subscribers_active {len(subscribers)}",
        "# HELP outpost_ring_buffer_depth Current depth of in-memory recent events buffer",
        "# TYPE outpost_ring_buffer_depth gauge",
        f"outpost_ring_buffer_depth {len(recent_events)}",
        "# HELP outpost_snapshot_storage_bytes Total disk space occupied by snapshots",
        "# TYPE outpost_snapshot_storage_bytes gauge",
        f"outpost_snapshot_storage_bytes {storage_bytes}",
        "# HELP outpost_snapshot_files_total Total count of snapshot JPEG files on disk",
        "# TYPE outpost_snapshot_files_total gauge",
        f"outpost_snapshot_files_total {snap_stats['total_files']}",
        "# HELP outpost_alerts_triggered_total Total high-priority target alerts recorded",
        "# TYPE outpost_alerts_triggered_total counter",
        f"outpost_alerts_triggered_total {len(recent_alerts)}",
    ]

    for sp, info in species_stats.items():
        lines.append(f'outpost_species_sightings_total{{species="{sp}"}} {info["count"]}')

    for sid, st in stream_telemetry.items():
        val = 1 if st.get("status") == "active" else 0
        lines.append(f'outpost_stream_status{{stream_id="{sid}"}} {val}')

    output = "\n".join(lines) + "\n"
    return Response(content=output, media_type="text/plain; version=0.0.4")



@app.get("/events/recent", response_model=List[DetectionEvent])
async def get_recent_events(limit: int = 50):
    """Retrieve recent detection events from the ring buffer."""
    events = list(recent_events)
    return events[-limit:]


@app.get("/api/streams")
async def get_streams():
    """Retrieve all monitored streams and their live telemetry status."""
    return list(stream_telemetry.values())


@app.get("/api/streams/watchdog")
async def get_watchdog_status():
    """Retrieve stream health and failover status from the watchdog."""
    return stream_watchdog.get_watchdog_status()


@app.post("/api/streams/{stream_id}/resolve")
async def resolve_stream(stream_id: str, quality: str = "720p", force: bool = False):
    """Triggers on-demand streamlink resolution with synthetic failover."""
    try:
        res = stream_watchdog.resolve_stream_url(stream_id, quality=quality, force=force)
        return res
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))



@app.get("/api/stats/species")
async def get_species_stats():
    """Retrieve species detection counts, peak confidence, and recency."""
    return sorted(species_stats.values(), key=lambda x: x["count"], reverse=True)


@app.get("/api/alerts")
async def get_alerts():
    """List configured target species alert rules."""
    return alert_rules


@app.post("/api/alerts", status_code=201)
async def create_alert(alert: AlertRule):
    """Register a new target species alert rule."""
    alert_rules.append(alert)
    return {"status": "created", "rule": alert}


@app.get("/api/alerts/recent")
async def get_recent_alerts(limit: int = 20):
    """Retrieve recent triggered species alerts."""
    return list(recent_alerts)[-limit:]


@app.get("/api/alerts/webhooks")
async def get_webhook_history(limit: int = 20):
    """Retrieve recent webhook dispatch history."""
    return alert_dispatcher.dispatch_history[-limit:]


@app.post("/api/events", status_code=201)
async def ingest_event(event: DetectionEvent):
    """Ingest a detection event from the 4090 CV runner and broadcast to SSE subscribers."""
    global total_events_dispatched

    # Update stream state
    if event.stream_id not in stream_telemetry:
        stream_telemetry[event.stream_id] = {
            "stream_id": event.stream_id,
            "name": event.stream_id,
            "provider": "Custom",
            "fps_target": 60,
            "status": "active",
            "total_detections": 0,
            "last_detection": None,
        }
    stream_telemetry[event.stream_id]["status"] = "active"
    stream_telemetry[event.stream_id]["total_detections"] += 1
    stream_telemetry[event.stream_id]["last_detection"] = event.timestamp

    # Update species aggregation stats
    sp = event.species
    if sp not in species_stats:
        species_stats[sp] = {
            "species": sp,
            "count": 0,
            "peak_confidence": event.confidence,
            "last_seen": event.timestamp,
            "streams": [event.stream_id],
        }
    species_stats[sp]["count"] += 1
    species_stats[sp]["peak_confidence"] = max(species_stats[sp]["peak_confidence"], event.confidence)
    species_stats[sp]["last_seen"] = event.timestamp
    if event.stream_id not in species_stats[sp]["streams"]:
        species_stats[sp]["streams"].append(event.stream_id)

    # Check alert rules
    for rule in alert_rules:
        if rule.enabled and rule.species.lower() == sp.lower():
            if (rule.stream_id is None or rule.stream_id == event.stream_id) and event.confidence >= rule.min_confidence:
                alert_entry = {
                    "alert_id": f"alt_{uuid.uuid4().hex[:8]}",
                    "rule_id": rule.rule_id,
                    "event_id": event.event_id,
                    "species": sp,
                    "confidence": event.confidence,
                    "stream_id": event.stream_id,
                    "timestamp": event.timestamp,
                }
                recent_alerts.append(alert_entry)
                event.metadata["alert_triggered"] = True
                event.metadata["matched_rule_id"] = rule.rule_id

                # Dispatch webhook via AlertDispatcher
                dispatch_res = alert_dispatcher.dispatch(
                    event.model_dump(),
                    stream_info=stream_telemetry.get(event.stream_id),
                )
                event.metadata["webhook_dispatch"] = dispatch_res
                break

    recent_events.append(event)
    total_events_dispatched += 1

    # Broadcast to all connected SSE clients
    payload = json.dumps(event.model_dump())
    dead_queues = set()
    for q in subscribers:
        try:
            q.put_nowait(payload)
        except asyncio.QueueFull:
            dead_queues.add(q)

    # Clean up any dead/overflowed queues
    for dq in dead_queues:
        subscribers.discard(dq)

    return {"status": "broadcasted", "event_id": event.event_id, "subscribers": len(subscribers)}


@app.get("/events")
async def sse_events(request: Request):
    """Server-Sent Events stream for connected clients."""
    client_queue: asyncio.Queue = asyncio.Queue(maxsize=100)
    subscribers.add(client_queue)

    async def event_generator():
        try:
            # Yield initial handshake / connection ack
            yield f"event: connected\ndata: {json.dumps({'message': 'Connected to Outpost SSE event bus', 'buffer_depth': len(recent_events)})}\n\n"

            while True:
                try:
                    # Wait for next event with a 15-second keepalive ping
                    payload = await asyncio.wait_for(client_queue.get(), timeout=15.0)
                    yield f"event: detection\ndata: {payload}\n\n"
                except asyncio.TimeoutError:
                    # Heartbeat comment to keep connection alive through proxies
                    yield ": ping\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            subscribers.discard(client_queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
