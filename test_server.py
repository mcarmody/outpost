"""Unit and integration tests for Outpost FastAPI SSE server."""

import asyncio
import json
import pytest
from fastapi.testclient import TestClient
from server import app, recent_events, subscribers, total_events_dispatched


@pytest.fixture(autouse=True)
def reset_state():
    """Reset in-memory state before each test."""
    recent_events.clear()
    subscribers.clear()


def test_health_check():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["service"] == "outpost-telemetry-bus"
    assert "active_sse_subscribers" in data
    assert "ring_buffer_depth" in data


def test_ingest_and_recent_events():
    client = TestClient(app)
    payload = {
        "event_id": "evt_test_01",
        "stream_id": "anacapa_kelp_01",
        "species": "garibaldi",
        "confidence": 0.92,
        "bbox": [100, 200, 150, 280],
        "metadata": {"water_temp_c": 18.5}
    }
    response = client.post("/api/events", json=payload)
    assert response.status_code == 201
    res_data = response.json()
    assert res_data["status"] == "broadcasted"
    assert res_data["event_id"] == "evt_test_01"

    # Verify ring buffer query
    rec_res = client.get("/events/recent")
    assert rec_res.status_code == 200
    events = rec_res.json()
    assert len(events) == 1
    assert events[0]["species"] == "garibaldi"
    assert events[0]["confidence"] == 0.92


def test_ring_buffer_cap():
    client = TestClient(app)
    for i in range(250):
        client.post("/api/events", json={
            "stream_id": "feeder_01",
            "species": f"bird_{i}",
            "confidence": 0.85,
            "bbox": [10, 10, 50, 50]
        })

    rec_res = client.get("/events/recent?limit=250")
    events = rec_res.json()
    # Ring buffer maxlen is 200
    assert len(events) == 200
    assert events[0]["species"] == "bird_50"
    assert events[-1]["species"] == "bird_249"


@pytest.mark.anyio
async def test_sse_subscriber_broadcast():
    client = TestClient(app)
    queue: asyncio.Queue = asyncio.Queue(maxsize=10)
    subscribers.add(queue)

    event_payload = {
        "event_id": "evt_broadcast_01",
        "stream_id": "feeder_cam",
        "species": "blue_jay",
        "confidence": 0.95,
        "bbox": [50, 60, 120, 180],
    }
    res = client.post("/api/events", json=event_payload)
    assert res.status_code == 201

    # Verify message was broadcast to subscriber queue
    msg_str = await queue.get()
    msg_data = json.loads(msg_str)
    assert msg_data["event_id"] == "evt_broadcast_01"
    assert msg_data["species"] == "blue_jay"
    subscribers.discard(queue)


def test_simulate_event_integration():
    from simulate import simulate_event
    client = TestClient(app)
    payload = simulate_event(api_url=None)
    assert payload["species"]
    assert payload["snapshot_url"].startswith("/snapshots/")

    # Ingest into server
    resp = client.post("/api/events", json=payload)
    assert resp.status_code == 201

    # Verify snapshot static endpoint returns the image
    snap_resp = client.get(payload["snapshot_url"])
    assert snap_resp.status_code == 200
    assert "image" in snap_resp.headers.get("content-type", "")
