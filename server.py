"""FastAPI Server-Sent Events (SSE) & Telemetry Dispatcher for Project Outpost.

Provides:
- GET /events: Real-time SSE stream for web UI clients
- GET /events/recent: Ring-buffer snapshot for immediate client state
- POST /api/events: Ingest endpoint for wildlife_cv_runner (4090 / CUDA)
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

SNAPSHOTS_DIR = Path("/workspace/scratch/outpost/snapshots")
SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
INDEX_HTML = Path("/workspace/scratch/outpost/index.html")

app = FastAPI(
    title="Outpost Wildlife CV Telemetry Bus",
    description="Real-time SSE event dispatcher and ring buffer for public wildlife livestreams",
    version="0.1.0",
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


# In-memory ring buffer of recent events (depth: 200)
recent_events: deque = deque(maxlen=200)
subscribers: Set[asyncio.Queue] = set()
total_events_dispatched = 0
start_time = time.time()


@app.get("/health")
async def health_check():
    """System telemetry and active subscriber metrics."""
    return {
        "status": "online",
        "service": "outpost-telemetry-bus",
        "uptime_seconds": round(time.time() - start_time, 2),
        "active_sse_subscribers": len(subscribers),
        "ring_buffer_depth": len(recent_events),
        "total_events_dispatched": total_events_dispatched,
    }


@app.get("/events/recent", response_model=List[DetectionEvent])
async def get_recent_events(limit: int = 50):
    """Retrieve recent detection events from the ring buffer."""
    events = list(recent_events)
    return events[-limit:]


@app.post("/api/events", status_code=201)
async def ingest_event(event: DetectionEvent):
    """Ingest a detection event from the 4090 CV runner and broadcast to SSE subscribers."""
    global total_events_dispatched
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
