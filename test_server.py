"""Unit and integration tests for Outpost FastAPI SSE server."""

import asyncio
import json
import pytest
from fastapi.testclient import TestClient
from server import (
    app,
    recent_events,
    subscribers,
    species_stats,
    stream_telemetry,
    recent_alerts,
    alert_rules,
    AlertRule,
)


@pytest.fixture(autouse=True)
def reset_state():
    """Reset in-memory state before each test."""
    recent_events.clear()
    subscribers.clear()
    species_stats.clear()
    recent_alerts.clear()


def test_health_check():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["service"] == "outpost-telemetry-bus"
    assert "active_sse_subscribers" in data
    assert "ring_buffer_depth" in data


def test_root_dashboard():
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "Project Outpost" in response.text


def test_ingest_and_recent_events():
    client = TestClient(app)
    payload = {
        "event_id": "evt_test_01",
        "stream_id": "anacapa_kelp_01",
        "species": "garibaldi",
        "confidence": 0.92,
        "bbox": [100, 200, 150, 280],
        "metadata": {"water_temp_c": 18.5},
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


def test_streams_and_species_stats():
    client = TestClient(app)
    payload1 = {
        "stream_id": "anacapa_kelp_01",
        "species": "giant_sea_bass",
        "confidence": 0.88,
        "bbox": [50, 50, 200, 200],
    }
    payload2 = {
        "stream_id": "anacapa_kelp_01",
        "species": "giant_sea_bass",
        "confidence": 0.94,
        "bbox": [60, 60, 210, 210],
    }
    client.post("/api/events", json=payload1)
    client.post("/api/events", json=payload2)

    # Verify streams endpoint
    streams_res = client.get("/api/streams")
    assert streams_res.status_code == 200
    streams = streams_res.json()
    anacapa = next(s for s in streams if s["stream_id"] == "anacapa_kelp_01")
    assert anacapa["status"] == "active"
    assert anacapa["total_detections"] >= 2

    # Verify species stats
    stats_res = client.get("/api/stats/species")
    assert stats_res.status_code == 200
    stats = stats_res.json()
    assert len(stats) >= 1
    bass = next(s for s in stats if s["species"] == "giant_sea_bass")
    assert bass["count"] == 2
    assert bass["peak_confidence"] == 0.94


def test_alert_rules_and_triggering():
    client = TestClient(app)
    # Register alert rule for bald_eagle
    rule_payload = {
        "species": "bald_eagle",
        "min_confidence": 0.75,
        "enabled": True,
    }
    create_res = client.post("/api/alerts", json=rule_payload)
    assert create_res.status_code == 201

    # Ingest non-matching event
    client.post("/api/events", json={
        "stream_id": "cornell_feeder_01",
        "species": "chickadee",
        "confidence": 0.85,
        "bbox": [10, 10, 20, 20],
    })
    assert len(recent_alerts) == 0

    # Ingest matching event
    eagle_resp = client.post("/api/events", json={
        "stream_id": "cornell_feeder_01",
        "species": "bald_eagle",
        "confidence": 0.91,
        "bbox": [50, 50, 300, 300],
    })
    assert eagle_resp.status_code == 201
    assert len(recent_alerts) == 1
    assert recent_alerts[0]["species"] == "bald_eagle"

    # Query recent alerts endpoint
    alerts_res = client.get("/api/alerts/recent")
    assert alerts_res.status_code == 200
    alerts_data = alerts_res.json()
    assert len(alerts_data) == 1
    assert alerts_data[0]["species"] == "bald_eagle"


def test_ring_buffer_cap():
    client = TestClient(app)
    for i in range(250):
        client.post("/api/events", json={
            "stream_id": "feeder_01",
            "species": f"bird_{i}",
            "confidence": 0.85,
            "bbox": [10, 10, 50, 50],
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


def test_retention_and_prune(tmp_path):
    from retention import get_snapshots_storage_stats, prune_snapshots

    # Create test snapshots in tmp_path
    f1 = tmp_path / "snap1.jpg"
    f2 = tmp_path / "snap2.jpg"
    f1.write_bytes(b"0" * 1024 * 500)  # 500 KB
    f2.write_bytes(b"0" * 1024 * 500)  # 500 KB

    stats = get_snapshots_storage_stats(tmp_path)
    assert stats["total_files"] == 2
    assert stats["total_mb"] >= 0.9

    # Prune with ceiling smaller than 1MB
    res = prune_snapshots(tmp_path, max_age_hours=24.0, max_storage_mb=0.6)
    assert res["pruned_files"] >= 1
    assert res["freed_mb"] > 0


def test_maintenance_prune_endpoint():
    client = TestClient(app)
    resp = client.post("/api/maintenance/prune?max_age_hours=48.0&max_storage_mb=500.0")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert "pruned_files" in data
    assert "freed_mb" in data


def test_webhook_alert_dispatch():
    client = TestClient(app)
    from server import alert_dispatcher

    alert_dispatcher.last_dispatched.clear()
    alert_dispatcher.dispatch_history.clear()

    # Trigger alert for bald_eagle
    res = client.post("/api/events", json={
        "stream_id": "cornell_feeder_01",
        "species": "bald_eagle",
        "confidence": 0.95,
        "bbox": [100, 100, 400, 400],
    })
    assert res.status_code == 201

    # Check webhook history endpoint
    hist_res = client.get("/api/alerts/webhooks")
    assert hist_res.status_code == 200
    history = hist_res.json()
    assert len(history) == 1
    assert history[0]["species"] == "bald_eagle"
    assert history[0]["delivered"] is True

    # Immediate second detection of same species should be rate-limited
    res2 = client.post("/api/events", json={
        "stream_id": "cornell_feeder_01",
        "species": "bald_eagle",
        "confidence": 0.96,
        "bbox": [105, 105, 410, 410],
    })
    assert res2.status_code == 201
    # History length should still be 1 because it was rate-limited
    hist_res2 = client.get("/api/alerts/webhooks")
    assert len(hist_res2.json()) == 1


