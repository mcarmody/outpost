"""Unit and integration tests for Outpost FastAPI SSE server."""

import asyncio
import json
import os
import time
import pytest

# Ensure isolated SQLite database per test process to prevent cross-run interference
os.environ["OUTPOST_DB_PATH"] = f"/tmp/outpost_test_{os.getpid()}.db"

from fastapi.testclient import TestClient
import db
from server import (
    app,
    recent_events,
    subscribers,
    species_stats,
    stream_telemetry,
    recent_alerts,
    alert_rules,
    AlertRule,
    SNAPSHOTS_DIR,
    hydrate_from_db,
)


@pytest.fixture(autouse=True)
def reset_state():
    """Reset in-memory and database state before each test."""
    db.clear_events()
    recent_events.clear()
    subscribers.clear()
    species_stats.clear()
    recent_alerts.clear()
    for st in stream_telemetry.values():
        st["total_detections"] = 0
        st["status"] = "idle"
        st["last_detection"] = None


def test_health_check():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["service"] == "outpost-telemetry-bus"
    assert "active_sse_subscribers" in data
    assert "ring_buffer_depth" in data
    assert "database" in data
    assert "total_events" in data["database"]


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
        "species": "person",
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
    assert events[0]['species'] == 'person'
    assert events[0]["confidence"] == 0.92


def test_streams_and_species_stats():
    client = TestClient(app)
    payload1 = {
        "stream_id": "anacapa_kelp_01",
        "species": "person",
        "confidence": 0.88,
        "bbox": [50, 50, 200, 200],
    }
    payload2 = {
        "stream_id": "anacapa_kelp_01",
        "species": "person",
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
    bass = next(s for s in stats if s["species"] == "person")
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

    # simulate.py's own species roster (flavor names for the synthetic
    # renderer, e.g. "Garibaldi") is intentionally cosmetic and independent
    # of server.py's real-world per-stream allowlists — some draws
    # legitimately get filtered (200, not 201). Found live 2026-09-21: this
    # made the test flaky, and overriding species alone wasn't enough
    # either — simulate_event() also randomizes stream_id, and "person"
    # only clears anacapa_kelp_01's allowlist, not Cornell's or Katmai's.
    # Pin both so this test verifies ingestion mechanics deterministically,
    # independent of allowlist/simulator drift on any given draw.
    payload["stream_id"] = "anacapa_kelp_01"
    payload["species"] = "person"

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


def test_stream_watchdog_and_failover():
    client = TestClient(app)
    from server import stream_watchdog

    # Check watchdog status endpoint
    resp = client.get("/api/streams/watchdog")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) >= 3
    streams = {s["stream_id"]: s for s in data}
    assert "anacapa_kelp_01" in streams
    assert "embed_url" in streams["anacapa_kelp_01"]
    assert "OAJF1Ie1m_Q" in streams["anacapa_kelp_01"]["embed_url"]

    # Test katmai alias registration in watchdog
    assert "katmai_brooks_falls" in streams

    # Register a synthetic/mock stream that triggers failover
    stream_watchdog.register_stream("test_mock_stream", "mock://offline_url", "Test Provider")
    res = stream_watchdog.resolve_stream_url("test_mock_stream")
    assert res["mode"] == "fallback_synthetic"

    # Trigger resolution via API endpoint
    api_res = client.post("/api/streams/test_mock_stream/resolve")
    assert api_res.status_code == 200
    api_data = api_res.json()
    assert api_data["mode"] == "fallback_synthetic"


def test_stream_embed_endpoint():
    client = TestClient(app)

    # Anacapa embed endpoint
    resp = client.get("/api/streams/anacapa_kelp_01/embed")
    assert resp.status_code == 200
    data = resp.json()
    assert data["stream_id"] == "anacapa_kelp_01"
    assert "embed_url" in data
    assert "youtube-nocookie.com/embed/OAJF1Ie1m_Q" in data["embed_url"]

    # Katmai alias embed endpoint
    resp_alias = client.get("/api/streams/katmai_brooks_falls/embed")
    assert resp_alias.status_code == 200
    data_alias = resp_alias.json()
    assert data_alias["stream_id"] == "katmai_brooks_falls"
    assert "J7ZrIDvqlic" in data_alias["embed_url"]

    # Unknown stream returns 404
    resp_err = client.get("/api/streams/unknown_stream_99/embed")
    assert resp_err.status_code == 404

    # Full /api/streams endpoint exposes embed_url
    streams_resp = client.get("/api/streams")
    assert streams_resp.status_code == 200
    for s in streams_resp.json():
        assert "embed_url" in s
        assert "watch_url" in s


def test_prometheus_metrics():
    client = TestClient(app)
    # Ingest event to populate counters
    client.post("/api/events", json={
        "stream_id": "anacapa_kelp_01",
        "species": "person",
        "confidence": 0.92,
        "bbox": [10, 10, 50, 50],
    })

    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "text/plain" in resp.headers["content-type"]
    body = resp.text
    assert "outpost_uptime_seconds" in body
    assert "outpost_events_dispatched_total" in body
    assert "outpost_snapshot_storage_bytes" in body
    assert 'outpost_species_sightings_total{species="person"}' in body


def test_hardware_diagnostics_and_supervisor():
    client = TestClient(app)
    resp = client.get("/api/supervisor")
    assert resp.status_code == 200
    data = resp.json()
    assert "cuda_available" in data
    assert "device_name" in data
    assert "recommended_mode" in data

    # Verify health response includes hardware section
    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    health_data = health_resp.json()
    assert "hardware" in health_data
    assert "cuda_available" in health_data["hardware"]


def test_stream_species_allowlist_filtering():
    client = TestClient(app)

    # 1. Allowed species on cornell_feeder_01
    allowed_resp = client.post("/api/events", json={
        "stream_id": "cornell_feeder_01",
        "species": "Northern Cardinal",
        "confidence": 0.88,
        "bbox": [10, 10, 50, 50],
    })
    assert allowed_resp.status_code == 201
    assert allowed_resp.json()["status"] == "broadcasted"

    # 2. Out-of-domain generic COCO hallucination (e.g. bear on a feeder) -> filtered
    filtered_resp = client.post("/api/events", json={
        "stream_id": "cornell_feeder_01",
        "species": "bear",
        "confidence": 0.78,
        "bbox": [10, 10, 50, 50],
    })
    assert filtered_resp.status_code == 200
    f_data = filtered_resp.json()
    assert f_data["status"] == "filtered"
    assert f_data["filtered"] is True
    assert "filtered by allowlist" in f_data["reason"]

    # 3. But bear on katmai_brooks_01 IS allowed
    katmai_resp = client.post("/api/events", json={
        "stream_id": "katmai_brooks_01",
        "species": "brown_bear",
        "confidence": 0.85,
        "bbox": [10, 10, 50, 50],
    })
    assert katmai_resp.status_code == 201
    assert katmai_resp.json()["status"] == "broadcasted"

    # Verify ring buffer only contains the 2 allowed events
    rec_res = client.get("/events/recent")
    events = rec_res.json()
    assert len(events) == 2
    assert {e["species"] for e in events} == {"Northern Cardinal", "brown_bear"}


def test_allowlist_management_endpoints():
    client = TestClient(app)

    # Fetch all allowlists
    resp = client.get("/api/streams/allowlists")
    assert resp.status_code == 200
    assert "cornell_feeder_01" in resp.json()

    # Fetch single stream allowlist
    resp2 = client.get("/api/streams/cornell_feeder_01/allowlist")
    assert resp2.status_code == 200
    assert "bird" in resp2.json()["allowed_species"]

    # Update allowlist
    resp3 = client.post("/api/streams/custom_stream_99/allowlist", json={
        "allowed_species": ["hawk", "falcon"],
        "strict_filtering": True,
    })
    assert resp3.status_code == 200
    assert resp3.json()["allowed_species"] == ["hawk", "falcon"]


def test_snapshot_upload_endpoint():
    client = TestClient(app)
    dummy_jpeg = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00\xff\xdb"

    files = {"file": ("test_frame.jpg", dummy_jpeg, "image/jpeg")}
    data = {"event_id": "evt_upload_123", "stream_id": "cornell_feeder_01"}
    resp = client.post("/api/snapshots/upload", files=files, data=data)
    assert resp.status_code == 200
    res_data = resp.json()
    assert res_data["status"] == "uploaded"
    assert "evt_upload_123.jpg" in res_data["snapshot_url"]
    assert res_data["size_bytes"] == len(dummy_jpeg)


def test_multipart_event_upload_endpoint():
    client = TestClient(app)
    dummy_jpeg = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00\xff\xdb"
    event_payload = {
        "event_id": "evt_multipart_456",
        "stream_id": "cornell_feeder_01",
        "species": "Blue Jay",
        "confidence": 0.94,
        "bbox": [20, 20, 100, 100],
    }

    files = {"file": ("jay.jpg", dummy_jpeg, "image/jpeg")}
    data = {"event_data": json.dumps(event_payload)}
    resp = client.post("/api/events/upload", files=files, data=data)
    assert resp.status_code == 201
    res_data = resp.json()
    assert res_data["status"] == "broadcasted"
    assert "evt_multipart_456.jpg" in res_data["snapshot_url"]

    # Verify recent events has the uploaded snapshot URL
    rec = client.get("/events/recent").json()
    assert any(e["event_id"] == "evt_multipart_456" and "evt_multipart_456.jpg" in e["snapshot_url"] for e in rec)

    # Disallowed species via multipart upload (e.g. bear on cornell_feeder_01)
    filtered_payload = {
        "event_id": "evt_multipart_bear",
        "stream_id": "cornell_feeder_01",
        "species": "bear",
        "confidence": 0.78,
        "bbox": [20, 20, 100, 100],
    }
    f_files = {"file": ("bear.jpg", dummy_jpeg, "image/jpeg")}
    f_data = {"event_data": json.dumps(filtered_payload)}
    f_resp = client.post("/api/events/upload", files=f_files, data=f_data)
    assert f_resp.status_code == 200
    f_res_data = f_resp.json()
    assert f_res_data["status"] == "filtered"
    assert f_res_data["filtered"] is True
    assert "filtered by allowlist" in f_res_data["reason"]

    # Verify filtered snapshot was NOT persisted to disk
    today_str = time.strftime("%Y%m%d")
    bear_snapshot = SNAPSHOTS_DIR / today_str / "evt_multipart_bear.jpg"
    assert not bear_snapshot.exists()


def test_runner_edge_allowlist_filtering(monkeypatch):
    """Verify runner.py client-side allowlist caching and edge pre-filtering (Issue #2)."""
    from runner import _cached_allowlists, is_species_allowed_edge
    _cached_allowlists.clear()
    client = TestClient(app)

    def mock_get(url, timeout=2.0):
        # Extract endpoint path
        path = "/" + url.split("://", 1)[-1].split("/", 1)[-1]
        return client.get(path)

    monkeypatch.setattr("runner.requests.get", mock_get)

    # Cornell feeder allows Northern Cardinal, suppresses bear
    assert is_species_allowed_edge("http://testserver", "cornell_feeder_01", "Northern Cardinal") is True
    assert is_species_allowed_edge("http://testserver", "cornell_feeder_01", "bear") is False

    # Katmai allows brown bear
    assert is_species_allowed_edge("http://testserver", "katmai_brooks_01", "brown_bear") is True


def test_mobile_shell_invariants():
    """Verify mobile navigation, Field Station identity, and responsive tab layout in index.html."""
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    html = resp.text

    assert "PROJECT OUTPOST" in html
    assert "Field Station" in html
    assert "setMobileTab" in html
    assert "tab-btn-streams" in html
    assert "tab-btn-sightings" in html
    assert "tab-btn-station" in html
    assert "aspect-video" in html
    assert "tile-anacapa_kelp_01" in html
    assert "tile-cornell_feeder_01" in html
    assert "tile-katmai_brooks_falls" in html


def test_stream_allowlist_alias_resolution_and_simulation():
    """Verify stream allowlist endpoint resolves aliases and simulate.py provides in-domain species."""
    from simulate import STREAMS
    client = TestClient(app)

    # 1. Alias resolution on GET /api/streams/{stream_id}/allowlist
    resp_canonical = client.get("/api/streams/katmai_brooks_01/allowlist")
    assert resp_canonical.status_code == 200
    data_canonical = resp_canonical.json()
    assert "bear" in data_canonical["allowed_species"]

    resp_alias = client.get("/api/streams/katmai_brooks_falls/allowlist")
    assert resp_alias.status_code == 200
    data_alias = resp_alias.json()
    assert "bear" in data_alias["allowed_species"]

    # 2. Verify simulate.py provides in-domain species for Anacapa
    anacapa_species = [s[0] for s in STREAMS["anacapa_kelp_01"]["species"]]
    assert any("person" in s.lower() or "diver" in s.lower() for s in anacapa_species)
    assert any("boat" in s.lower() or "vessel" in s.lower() for s in anacapa_species)


def test_events_recent_filtering():
    """Verify /events/recent filters by stream_id and species."""
    client = TestClient(app)
    # Ingest event 1 (allowed on anacapa)
    client.post("/api/events", json={
        "event_id": "evt_filter_01",
        "stream_id": "anacapa_kelp_01",
        "species": "boat",
        "confidence": 0.91,
        "bbox": [10, 20, 30, 40],
    })
    # Ingest event 2 (allowed on cornell)
    client.post("/api/events", json={
        "event_id": "evt_filter_02",
        "stream_id": "cornell_feeder_01",
        "species": "blue_jay",
        "confidence": 0.85,
        "bbox": [15, 25, 35, 45],
    })
    # Ingest event 3 (allowed on anacapa)
    client.post("/api/events", json={
        "event_id": "evt_filter_03",
        "stream_id": "anacapa_kelp_01",
        "species": "person",
        "confidence": 0.95,
        "bbox": [50, 60, 70, 80],
    })

    # All events
    res_all = client.get("/events/recent")
    assert res_all.status_code == 200
    assert len(res_all.json()) == 3

    # Filter by stream_id
    res_anacapa = client.get("/events/recent?stream_id=anacapa_kelp_01")
    assert res_anacapa.status_code == 200
    assert len(res_anacapa.json()) == 2
    assert all(e["stream_id"] == "anacapa_kelp_01" for e in res_anacapa.json())

    # Filter by species
    res_species = client.get("/events/recent?species=blue_jay")
    assert res_species.status_code == 200
    assert len(res_species.json()) == 1
    assert res_species.json()[0]["species"] == "blue_jay"


def test_get_event_by_id():
    """Verify GET /api/events/{event_id} retrieves event or returns 404."""
    client = TestClient(app)
    client.post("/api/events", json={
        "event_id": "evt_lookup_123",
        "stream_id": "anacapa_kelp_01",
        "species": "person",
        "confidence": 0.88,
        "bbox": [10, 20, 30, 40],
    })

    # Lookup existing
    res_ok = client.get("/api/events/evt_lookup_123")
    assert res_ok.status_code == 200
    assert res_ok.json()["event_id"] == "evt_lookup_123"
    assert res_ok.json()["species"] == "person"

    # Lookup nonexistent
    res_404 = client.get("/api/events/evt_nonexistent_999")
    assert res_404.status_code == 404
    assert "not found" in res_404.json()["detail"].lower()


def test_species_stats_stream_filtering():
    """Verify /api/stats/species filters stats by stream_id."""
    client = TestClient(app)
    client.post("/api/events", json={
        "event_id": "evt_stat_01",
        "stream_id": "anacapa_kelp_01",
        "species": "person",
        "confidence": 0.88,
        "bbox": [10, 20, 30, 40],
    })
    client.post("/api/events", json={
        "event_id": "evt_stat_02",
        "stream_id": "cornell_feeder_01",
        "species": "blue_jay",
        "confidence": 0.90,
        "bbox": [10, 20, 30, 40],
    })

    # Unfiltered
    res_all = client.get("/api/stats/species")
    assert res_all.status_code == 200
    assert len(res_all.json()) == 2

    # Filtered by stream
    res_anacapa = client.get("/api/stats/species?stream_id=anacapa_kelp_01")
    assert res_anacapa.status_code == 200
    assert len(res_anacapa.json()) == 1
    assert res_anacapa.json()[0]["species"] == "person"


def test_index_html_lightbox_and_quick_filters():
    """Verify index.html contains lightbox modal, stream filters, and dynamic select support."""
    client = TestClient(app)
    res = client.get("/")
    assert res.status_code == 200
    html = res.text

    assert "snapshot-modal" in html
    assert "feed-filter-all" in html
    assert "feed-filter-anacapa" in html
    assert "openEventModal" in html
    assert "openStreamSnapshotModal" in html
    assert "ensureSpeciesInDropdown" in html


def test_sqlite_persistence_and_restart_survivability():
    """Verify events persist in SQLite and can be hydrated back after server restart."""
    client = TestClient(app)

    # Ingest 2 events
    evt1 = {
        "event_id": "evt_survive_01",
        "stream_id": "anacapa_kelp_01",
        "species": "person",
        "confidence": 0.91,
        "bbox": [10, 20, 30, 40],
    }
    evt2 = {
        "event_id": "evt_survive_02",
        "stream_id": "cornell_feeder_01",
        "species": "blue_jay",
        "confidence": 0.88,
        "bbox": [50, 60, 70, 80],
    }
    res1 = client.post("/api/events", json=evt1)
    res2 = client.post("/api/events", json=evt2)
    assert res1.status_code == 201
    assert res2.status_code == 201

    # Verify SQLite recorded both events
    db_stats = db.get_db_stats()
    assert db_stats["total_events"] == 2

    # Simulate server crash/restart: wipe in-memory structures completely
    recent_events.clear()
    species_stats.clear()
    for st in stream_telemetry.values():
        st["total_detections"] = 0
        st["status"] = "idle"
        st["last_detection"] = None

    assert len(recent_events) == 0

    # 1. Direct event lookup survives memory wipe via SQLite fallback
    lookup_res = client.get("/api/events/evt_survive_01")
    assert lookup_res.status_code == 200
    assert lookup_res.json()["event_id"] == "evt_survive_01"
    assert lookup_res.json()["species"] == "person"

    # 2. Server restart hydration restores recent_events, stream telemetry, and species stats
    hydrate_from_db()
    assert len(recent_events) == 2
    assert "person" in species_stats
    assert "blue_jay" in species_stats

    # 3. GET /events/recent returns hydrated events
    rec_res = client.get("/events/recent")
    assert rec_res.status_code == 200
    events = rec_res.json()
    assert len(events) == 2
    assert events[0]["event_id"] == "evt_survive_01"
    assert events[1]["event_id"] == "evt_survive_02"


def test_database_stats_endpoint():
    """Verify /api/database/stats exposes accurate record counts and file footprint."""
    client = TestClient(app)
    client.post("/api/events", json={
        "event_id": "evt_db_stat_01",
        "stream_id": "cornell_feeder_01",
        "species": "blue_jay",
        "confidence": 0.95,
        "bbox": [10, 20, 30, 40],
    })

    res = client.get("/api/database/stats")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ready"
    assert data["total_events"] >= 1
    assert "db_size_bytes" in data
    assert "db_size_mb" in data
    assert "oldest_event" in data
    assert "newest_event" in data


def test_database_prometheus_metrics():
    """Verify /metrics exposes SQLite total events and size metrics."""
    client = TestClient(app)
    client.post("/api/events", json={
        "event_id": "evt_prom_db_01",
        "stream_id": "cornell_feeder_01",
        "species": "cardinal",
        "confidence": 0.89,
        "bbox": [15, 25, 35, 45],
    })

    res = client.get("/metrics")
    assert res.status_code == 200
    text = res.text
    assert "outpost_database_events_total" in text
    assert "outpost_database_size_bytes" in text


def test_database_event_pruning():
    """Verify db.prune_events removes expired events older than cutoff."""
    # Insert an event timestamped 40 days ago
    old_time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - (40 * 86400)))
    db.save_event({
        "event_id": "evt_ancient_01",
        "timestamp": old_time,
        "stream_id": "cornell_feeder_01",
        "species": "cardinal",
        "confidence": 0.85,
        "bbox": [0, 0, 10, 10],
    })
    # Insert a fresh event
    db.save_event({
        "event_id": "evt_fresh_01",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stream_id": "cornell_feeder_01",
        "species": "cardinal",
        "confidence": 0.92,
        "bbox": [0, 0, 10, 10],
    })

    assert db.get_event("evt_ancient_01") is not None
    assert db.get_event("evt_fresh_01") is not None

    pruned = db.prune_events(max_age_days=30.0)
    assert pruned == 1
    assert db.get_event("evt_ancient_01") is None
    assert db.get_event("evt_fresh_01") is not None








