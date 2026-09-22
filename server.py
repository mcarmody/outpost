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
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from fastapi import FastAPI, HTTPException, Request, Response, UploadFile, File, Form
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

RETENTION_INTERVAL_SECONDS = 900.0  # 15 min
RETENTION_MAX_AGE_HOURS = 24.0
RETENTION_MAX_STORAGE_MB = 500.0


async def _retention_loop():
    """Background snapshot pruning, actually wired into the app lifecycle.

    This was claimed done ('wiring the automated background cleanup loop
    into the FastAPI lifecycle') but never actually landed — only the
    manual /api/maintenance/prune endpoint existed. Confirmed live
    2026-09-21 during the hourly check-in: snapshot_storage_mb had grown
    to 649MB / 1535 files, well past the 500MB ceiling this was supposed
    to enforce automatically."""
    while True:
        try:
            res = prune_snapshots(
                SNAPSHOTS_DIR,
                max_age_hours=RETENTION_MAX_AGE_HOURS,
                max_storage_mb=RETENTION_MAX_STORAGE_MB,
            )
            if res.get("pruned_files"):
                print(f"[retention] pruned {res['pruned_files']} files, "
                      f"freed {res['freed_mb']}MB, remaining {res['remaining_mb']}MB")
        except Exception as e:
            print(f"[retention] loop error (continuing): {e}")
        await asyncio.sleep(RETENTION_INTERVAL_SECONDS)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    task = asyncio.create_task(_retention_loop())
    try:
        yield
    finally:
        task.cancel()


app = FastAPI(
    title="Outpost Wildlife CV Telemetry Bus",
    description="Real-time SSE event dispatcher and ring buffer for public wildlife livestreams",
    version="0.2.0",
    lifespan=_lifespan,
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
    "anacapa_kelp_01": {
        "name": "Anacapa Island Kelp Forest",
        "provider": "Explore.org",
        "fps_target": 120,
        "youtube_id": "OAJF1Ie1m_Q",
        "embed_url": "https://www.youtube-nocookie.com/embed/OAJF1Ie1m_Q?autoplay=1&mute=1",
        "watch_url": "https://www.youtube.com/watch?v=OAJF1Ie1m_Q",
    },
    "cornell_feeder_01": {
        "name": "Cornell Lab FeederWatch",
        "provider": "Cornell Lab",
        "fps_target": 60,
        "youtube_id": "x10vL6_47Dw",
        "embed_url": "https://www.youtube-nocookie.com/embed/x10vL6_47Dw?autoplay=1&mute=1",
        "watch_url": "https://www.youtube.com/watch?v=x10vL6_47Dw",
    },
    "katmai_brooks_01": {
        "name": "Katmai Brooks Falls",
        "provider": "Explore.org",
        "fps_target": 60,
        "youtube_id": "J7ZrIDvqlic",
        "embed_url": "https://www.youtube-nocookie.com/embed/J7ZrIDvqlic?autoplay=1&mute=1",
        "watch_url": "https://www.youtube.com/watch?v=J7ZrIDvqlic",
    },
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

# Reported hardware from edge CV runners (e.g. Mike's RTX 4090 box), keyed
# by whatever they self-identify as. get_hardware_diagnostics() below only
# ever sees the relay server's own hardware (a CPU-only Linux box) — that
# made /health and the dashboard claim "no CUDA" even while real inference
# was running on the 4090, because nobody was asking the actual GPU box.
# runner.py registers here once at startup; entries older than
# EDGE_HARDWARE_TTL_SECONDS are treated as stale and ignored.
edge_runner_hardware: Dict[str, Dict[str, Any]] = {}
EDGE_HARDWARE_TTL_SECONDS = 180.0


def get_active_edge_hardware() -> Optional[Dict[str, Any]]:
    """Most recently registered, non-stale, CUDA-capable edge runner, if any."""
    now = time.time()
    candidates = [
        v for v in edge_runner_hardware.values()
        if now - v.get("reported_at", 0) <= EDGE_HARDWARE_TTL_SECONDS
    ]
    if not candidates:
        return None
    cuda_candidates = [c for c in candidates if c.get("cuda_available")]
    pool = cuda_candidates or candidates
    return max(pool, key=lambda c: c.get("reported_at", 0))

species_stats: Dict[str, Dict[str, Any]] = {}

# Stream-level species allowlists to suppress out-of-domain generic COCO hallucinations
stream_allowlists: Dict[str, Dict[str, Any]] = {
    "cornell_feeder_01": {
        "stream_id": "cornell_feeder_01",
        "allowed_species": [
            "bird", "mourning dove", "northern cardinal", "cardinal",
            "blue jay", "black-capped chickadee", "chickadee", "tufted titmouse",
            "titmouse", "white-breasted nuthatch", "nuthatch", "american goldfinch",
            "goldfinch", "house finch", "finch", "downy woodpecker", "woodpecker",
            "red-bellied woodpecker", "squirrel", "eastern gray squirrel",
            "chipmunk", "common raven", "raven", "american crow", "crow",
            "dark-eyed junco", "sparrow", "song sparrow", "european starling",
            "bald eagle", "eagle", "hawk", "cooper's hawk", "sharp-shinned hawk",
            "raptor",
        ],
        "strict_filtering": True,
    },
    "anacapa_kelp_01": {
        "stream_id": "anacapa_kelp_01",
        # yolo11x.pt is COCO-trained, and COCO's 80 classes include no fish,
        # seal, shark, or ray — a strict allowlist of marine species (the
        # original list here) can NEVER match anything the model actually
        # emits, so the tile stays permanently empty regardless of whether
        # the stream itself is healthy. Confirmed live 2026-09-21: real
        # detections on this feed are COCO-generic misreads ("frisbee",
        # "broccoli" — light/kelp shapes), all correctly filtered, leaving
        # zero events ever landing. Narrowed to the only COCO classes that
        # could plausibly and correctly appear here (a diver as "person", a
        # passing vessel as "boat") until a marine-aware model (BioCLIP /
        # a fine-tuned detector — tracked as Outpost issue #2) replaces
        # generic COCO for this stream specifically.
        "allowed_species": ["person", "boat"],
        "strict_filtering": True,
    },
    "katmai_brooks_01": {
        "stream_id": "katmai_brooks_01",
        "allowed_species": [
            "bear", "brown bear", "grizzly bear", "salmon", "sockeye salmon",
            "fish", "bald eagle", "eagle", "gull", "glaucous-winged gull",
            "raven", "wolf",
        ],
        "strict_filtering": True,
    },
    "katmai_brooks_falls": {
        "stream_id": "katmai_brooks_falls",
        "allowed_species": [
            "bear", "brown bear", "grizzly bear", "salmon", "sockeye salmon",
            "fish", "bald eagle", "eagle", "gull", "glaucous-winged gull",
            "raven", "wolf",
        ],
        "strict_filtering": True,
    },
}


def is_species_allowed(stream_id: str, species: str) -> bool:
    """Evaluate whether a detected species is permissible on the given stream."""
    if not stream_id or not species:
        return True
    cfg = stream_allowlists.get(stream_id)
    if not cfg or not cfg.get("strict_filtering", False):
        return True
    allowed_list = cfg.get("allowed_species", [])
    sp_norm = species.strip().lower().replace("_", " ")
    for a in allowed_list:
        a_norm = a.strip().lower().replace("_", " ")
        if a_norm in sp_norm or sp_norm in a_norm:
            return True
    return False


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
stream_watchdog.register_stream("katmai_brooks_falls", "https://www.youtube.com/watch?v=J7ZrIDvqlic", "Explore.org")

# In-memory ring buffer of recent events (depth: 200)
recent_events: deque = deque(maxlen=200)
subscribers: Set[asyncio.Queue] = set()
total_events_dispatched = 0
start_time = time.time()


class EdgeRunnerRegistration(BaseModel):
    stream_id: str
    cuda_available: bool = False
    device_name: str = "CPU Only"
    total_vram_gb: float = 0.0
    hostname: Optional[str] = None


@app.post("/api/runner/register")
async def register_edge_runner(reg: EdgeRunnerRegistration):
    """Edge CV runner (e.g. runner.py on Mike's 4090) self-reports its real
    hardware here. /health and /api/supervisor prefer this over the relay
    server's own (usually CPU-only) hardware when a recent report exists."""
    edge_runner_hardware[reg.stream_id] = {
        "cuda_available": reg.cuda_available,
        "device_name": reg.device_name,
        "total_vram_gb": reg.total_vram_gb,
        "hostname": reg.hostname,
        "reported_at": time.time(),
    }
    return {"status": "registered", "stream_id": reg.stream_id}


def _hardware_for_status_endpoints() -> Dict[str, Any]:
    edge = get_active_edge_hardware()
    if edge:
        return {
            "cuda_available": edge["cuda_available"],
            "device_name": edge["device_name"],
            "total_vram_gb": edge["total_vram_gb"],
            "recommended_mode": "cuda_tensorrt" if edge["cuda_available"] else "synthetic",
            "host_platform": edge.get("hostname") or "edge-runner",
            "python_version": None,
            "hardware_source": "edge_runner",
        }
    hw = get_hardware_diagnostics()
    hw["hardware_source"] = "relay_server"
    return hw


@app.get("/health")
async def health_check():
    """System telemetry and active subscriber metrics."""
    snap_stats = get_snapshots_storage_stats(SNAPSHOTS_DIR)
    hw = _hardware_for_status_endpoints()
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
    """Returns hardware capability diagnostics and recommended execution mode
    — the active edge runner's real hardware when one has reported in
    recently, otherwise the relay server's own (see hardware_source)."""
    return _hardware_for_status_endpoints()



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


@app.get("/api/streams/{stream_id}/embed")
async def get_stream_embed(stream_id: str):
    """Retrieve verified video embed player parameters and fallback links."""
    stream = stream_telemetry.get(stream_id)
    if not stream and stream_id == "katmai_brooks_falls":
        stream = stream_telemetry.get("katmai_brooks_01")

    if not stream:
        raise HTTPException(status_code=404, detail=f"Stream '{stream_id}' not found in telemetry registry.")

    return {
        "stream_id": stream_id,
        "name": stream.get("name"),
        "provider": stream.get("provider"),
        "youtube_id": stream.get("youtube_id"),
        "embed_url": stream.get("embed_url"),
        "watch_url": stream.get("watch_url"),
    }



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


def process_event(event: DetectionEvent) -> dict:
    """Internal event ingestion engine: enforces species allowlist, aggregates stats, and broadcasts via SSE."""
    global total_events_dispatched

    # Species allowlist check
    if not is_species_allowed(event.stream_id, event.species):
        return {
            "status": "filtered",
            "filtered": True,
            "event_id": event.event_id,
            "stream_id": event.stream_id,
            "species": event.species,
            "reason": f"Species '{event.species}' filtered by allowlist for stream '{event.stream_id}'",
        }

    # Update stream state
    if event.stream_id not in stream_telemetry:
        stream_telemetry[event.stream_id] = {
            "stream_id": event.stream_id,
            "name": event.stream_id,
            "provider": "Custom",
            "fps_target": 60,
            "youtube_id": None,
            "embed_url": None,
            "watch_url": None,
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


@app.post("/api/events", status_code=201)
async def ingest_event(event: DetectionEvent, response: Response):
    """Ingest a detection event from edge runners (4090/CUDA) and broadcast to SSE subscribers."""
    res = process_event(event)
    if res.get("status") == "filtered":
        response.status_code = 200
    return res


@app.post("/api/snapshots/upload")
async def upload_snapshot(
    file: UploadFile = File(...),
    event_id: Optional[str] = Form(None),
    stream_id: Optional[str] = Form(None),
):
    """Save an edge-captured snapshot JPEG to the Outpost static storage."""
    evt_id = event_id or f"evt_{uuid.uuid4().hex[:8]}"
    today_str = time.strftime("%Y%m%d")
    day_dir = SNAPSHOTS_DIR / today_str
    day_dir.mkdir(parents=True, exist_ok=True)

    ext = Path(file.filename or "frame.jpg").suffix.lower()
    if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
        ext = ".jpg"
    dest_path = day_dir / f"{evt_id}{ext}"

    content = await file.read()
    with open(dest_path, "wb") as f:
        f.write(content)

    rel_url = f"/snapshots/{today_str}/{evt_id}{ext}"
    return {
        "status": "uploaded",
        "event_id": evt_id,
        "stream_id": stream_id,
        "snapshot_url": rel_url,
        "size_bytes": len(content),
    }


@app.post("/api/events/upload", status_code=201)
async def ingest_event_with_snapshot(
    file: UploadFile = File(...),
    event_data: str = Form(...),
    response: Response = None,
):
    """Multipart ingestion: saves snapshot file and processes DetectionEvent atomically."""
    try:
        data = json.loads(event_data)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid event_data JSON: {e}")

    evt = DetectionEvent(**data)

    today_str = time.strftime("%Y%m%d")
    day_dir = SNAPSHOTS_DIR / today_str
    day_dir.mkdir(parents=True, exist_ok=True)
    ext = Path(file.filename or "frame.jpg").suffix.lower()
    if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
        ext = ".jpg"
    dest_path = day_dir / f"{evt.event_id}{ext}"

    content = await file.read()
    with open(dest_path, "wb") as f:
        f.write(content)

    evt.snapshot_url = f"/snapshots/{today_str}/{evt.event_id}{ext}"
    res = process_event(evt)
    res["snapshot_url"] = evt.snapshot_url
    if res.get("status") == "filtered" and response:
        response.status_code = 200
    return res


@app.get("/api/streams/allowlists")
async def get_stream_allowlists():
    """Retrieve all configured stream species allowlists."""
    return stream_allowlists


@app.get("/api/streams/{stream_id}/allowlist")
async def get_single_stream_allowlist(stream_id: str):
    """Retrieve species allowlist configuration for a specific stream."""
    if stream_id not in stream_allowlists:
        return {"stream_id": stream_id, "allowed_species": [], "strict_filtering": False}
    return stream_allowlists[stream_id]


@app.post("/api/streams/{stream_id}/allowlist")
async def update_stream_allowlist(stream_id: str, payload: Dict[str, Any]):
    """Update or register species allowlist for a stream."""
    allowed = payload.get("allowed_species", [])
    strict = payload.get("strict_filtering", True)
    stream_allowlists[stream_id] = {
        "stream_id": stream_id,
        "allowed_species": allowed,
        "strict_filtering": strict,
    }
    return stream_allowlists[stream_id]


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
