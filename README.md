# 🦅 Project Outpost — Wildlife & Reef Computer Vision Ops Room

[![Tests](https://img.shields.io/badge/tests-17%2F17%20passing-brightgreen)](#test-suite-verification)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com)
[![Model](https://img.shields.io/badge/YOLO-v11x%20Ultralytics-blue)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](#)

Project Outpost is an autonomous 24/7 computer vision surveillance and wildlife telemetry pipeline. Designed for high-throughput edge hardware (NVIDIA GeForce RTX 4090), Outpost continuously monitors public nature livestreams (kelp forests, feeder stations, brown bear falls), classifies species with high-precision temporal persistence, dispatches real-time Server-Sent Events (SSE), and serves a dark Ops Room surveillance dashboard.

---

## 🏗️ Architecture

```
[Public Livestreams (HLS/RTSP)] ──> [Stream Watchdog (Token Cache)]
                                           │
                                           ▼
[Synthetic Frame Generator] ──────> [CUDA YOLOv11x Runner]
                                           │
                                           ▼
                                [3-Frame Temporal Filter]
                                           │
                                           ▼
                              [FastAPI Telemetry Hub]
                                           │
         ┌───────────────────┬─────────────┴─────────────┬───────────────────┐
         ▼                   ▼                           ▼                   ▼
   [SSE Stream Bus]   [Ring Buffer (200)]      [Discord Alerts]     [Retention Pruner]
     /events            /events/recent            Webhook             /api/maintenance
         │                   │                           │                   │
         └───────────────────┴─────────────┬─────────────┴───────────────────┘
                                           ▼
                            [Dark Ops Room Frontend]
                             http://localhost:8000
```

---

## 🚀 Quickstart

### 1. Install Dependencies
```bash
pip install fastapi uvicorn pillow requests pytest anyio
```

### 2. Launch Telemetry Hub
```bash
python3 -m uvicorn server:app --host 0.0.0.0 --port 8000
```

### 3. Launch Synthetic Event Generator
In a second terminal, start continuous frame simulation across Anacapa Island, Cornell FeederWatch, and Katmai Falls:
```bash
python3 simulate.py --continuous --interval 2.5
```

### 4. Open the Ops Room
Navigate to [http://localhost:8000](http://localhost:8000) in any modern browser.

---

## 📡 API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Single-page Ops Room surveillance dashboard |
| `GET` | `/health` | Service health, ring buffer depth, active SSE subscribers |
| `GET` | `/events` | Real-time Server-Sent Events (SSE) telemetry stream |
| `GET` | `/events/recent` | Recent detections from the in-memory ring buffer (default: 50, max: 200) |
| `POST` | `/api/events` | Ingest detection event with bounding box and snapshot URL |
| `GET` | `/api/streams` | Registered stream metadata and uptime status |
| `POST` | `/api/streams/{id}/resolve` | Trigger watchdog stream token refresh/failover |
| `GET` | `/api/stats/species` | Aggregated species count and peak confidence scores |
| `GET` | `/api/alerts` | List configured automated notification rules |
| `POST` | `/api/alerts` | Register a new alert rule |
| `GET` | `/api/alerts/recent` | Recent triggered alerts ring buffer |
| `POST` | `/api/maintenance/prune` | Two-pass snapshot age and storage pruning |
| `GET` | `/metrics` | Prometheus metrics exposition |
| `GET` | `/api/supervisor` | CUDA / RTX 4090 hardware diagnostics |

---

## 🧪 Test Suite Verification

Run the full automated test suite:
```bash
pytest
```
Includes `test_server.py` (14 unit/integration tests) and `test_e2e_pipeline.py` (3 end-to-end pipeline lifecycle tests). **17/17 tests passing green**.

---

## 📄 Documentation

- [STANDUP_7AM_REPORT.md](STANDUP_7AM_REPORT.md) — Comprehensive overnight prototype sprint briefing and hardware transition plan.
- [SPEC.md](SPEC.md) — Original technical specification and CUDA performance requirements.
