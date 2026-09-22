"""End-to-End integration tests for Project Outpost wildlife CV telemetry pipeline.

Validates complete workflow:
1. Synthetic frame generation and JPEG snapshot rendering to disk.
2. Event ingestion via /api/events with bounding box coordinates and snapshot URL.
3. Live ring buffer querying via /events/recent.
4. Static file serving of JPEG snapshots via /snapshots/{date}/{event_id}.jpg.
5. Species aggregation statistics via /api/stats/species.
6. Prometheus metrics exposition via /metrics.
7. Automated alert rule evaluation via /api/alerts and /api/alerts/recent.
8. Snapshot retention pruning endpoint via /api/maintenance/prune.
"""

import os
import time
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

import db
from server import (
    app,
    recent_events,
    subscribers,
    species_stats,
    recent_alerts,
    alert_rules,
    SNAPSHOTS_DIR,
)
from simulate import generate_synthetic_snapshot, STREAMS


@pytest.fixture(autouse=True)
def clean_pipeline_state():
    """Reset server state before each integration test."""
    db.clear_events()
    recent_events.clear()
    subscribers.clear()
    species_stats.clear()
    recent_alerts.clear()
    alert_rules.clear()


def test_e2e_synthetic_generation_ingest_and_static_serving():
    client = TestClient(app)

    # 1. Generate realistic synthetic snapshot frame
    event_id = f"e2e_evt_{int(time.time())}"
    stream_id = "anacapa_kelp_01"
    species = "person"
    confidence = 0.94
    bbox = [120, 80, 260, 200]
    bg_color = STREAMS[stream_id]["bg_color"]
    tag_color = (255, 140, 0)

    rel_url = generate_synthetic_snapshot(
        event_id=event_id,
        stream_id=stream_id,
        species=species,
        confidence=confidence,
        bbox=bbox,
        bg_color=bg_color,
        tag_color=tag_color,
    )

    # Verify physical file existence
    today_str = time.strftime("%Y%m%d")
    expected_path = SNAPSHOTS_DIR / today_str / f"{event_id}.jpg"
    assert expected_path.exists(), f"Snapshot file not created at {expected_path}"
    assert expected_path.stat().st_size > 500, "Snapshot image is unexpectedly empty"

    # 2. Ingest detection event into Outpost telemetry bus
    payload = {
        "event_id": event_id,
        "stream_id": stream_id,
        "species": species,
        "confidence": confidence,
        "bbox": bbox,
        "snapshot_url": rel_url,
        "metadata": {
            "water_temp_c": 16.8,
            "depth_m": 8.5,
            "synthetic": True,
        },
    }

    ingest_resp = client.post("/api/events", json=payload)
    assert ingest_resp.status_code == 201
    ingest_data = ingest_resp.json()
    assert ingest_data["status"] == "broadcasted"
    assert ingest_data["event_id"] == event_id

    # 3. Verify event is in the recent ring buffer
    recent_resp = client.get("/events/recent")
    assert recent_resp.status_code == 200
    events = recent_resp.json()
    assert len(events) >= 1
    matched = next((e for e in events if e["event_id"] == event_id), None)
    assert matched is not None
    assert matched["species"] == species
    assert matched["snapshot_url"] == rel_url

    # 4. Verify snapshot image is served by FastAPI static files mount
    static_resp = client.get(rel_url)
    assert static_resp.status_code == 200
    assert "image/jpeg" in static_resp.headers["content-type"]
    assert len(static_resp.content) == expected_path.stat().st_size

    # 5. Verify species stats aggregated
    stats_resp = client.get("/api/stats/species")
    assert stats_resp.status_code == 200
    stats = stats_resp.json()
    matched_stat = next((s for s in stats if s["species"] == species), None)
    assert matched_stat is not None
    assert matched_stat["count"] >= 1
    assert matched_stat["peak_confidence"] >= confidence

    # 6. Verify Prometheus metrics incremented
    metrics_resp = client.get("/metrics")
    assert metrics_resp.status_code == 200
    metrics_text = metrics_resp.text
    assert "outpost_events_dispatched_total" in metrics_text
    assert f'species="{species}"' in metrics_text


def test_e2e_alert_rule_and_trigger_lifecycle():
    client = TestClient(app)

    # 1. Register high-priority alert rule for brown bears
    rule_payload = {
        "rule_id": "rule_bear_sighting",
        "stream_id": "katmai_brooks_falls",
        "species": "Brown Bear",
        "min_confidence": 0.80,
        "notify_channel": "wildlife-alerts",
    }
    create_rule_resp = client.post("/api/alerts", json=rule_payload)
    assert create_rule_resp.status_code == 201

    # 2. Ingest non-matching detection (wrong species)
    client.post(
        "/api/events",
        json={
            "stream_id": "katmai_brooks_falls",
            "species": "Sockeye Salmon",
            "confidence": 0.95,
            "bbox": [50, 50, 100, 100],
        },
    )
    alerts_res = client.get("/api/alerts/recent")
    assert len(alerts_res.json()) == 0

    # 3. Ingest matching detection meeting confidence threshold
    client.post(
        "/api/events",
        json={
            "stream_id": "katmai_brooks_falls",
            "species": "Brown Bear",
            "confidence": 0.88,
            "bbox": [150, 100, 320, 280],
            "metadata": {"weight_est_kg": 350},
        },
    )

    # 4. Verify alert was triggered and recorded in recent alerts
    alerts_res2 = client.get("/api/alerts/recent")
    assert alerts_res2.status_code == 200
    alerts = alerts_res2.json()
    assert len(alerts) == 1
    assert alerts[0]["rule_id"] == "rule_bear_sighting"
    assert alerts[0]["species"] == "Brown Bear"


def test_e2e_maintenance_prune_endpoint():
    client = TestClient(app)

    # Trigger maintenance prune endpoint with dry run
    resp = client.post("/api/maintenance/prune?max_age_hours=48.0&max_storage_mb=500.0")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert "pruned_files" in data
    assert "freed_mb" in data
