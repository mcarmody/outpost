# Project Outpost — Architecture Spec & Buildout Plan
**Date:** 2026-09-20 23:40 PT  
**Authors:** Zero & Amos  
**Repository:** `brockventures/outpost`  
**Status:** DRAFT / PROPOSED  

---

## 1. Executive Summary & Objective
**Project Outpost** is a 24/7 automated computer vision (CV) observation and alerting platform monitoring public nature, wildlife, and marine reef livestreams (Explore.org, Cornell Lab). It leverages local bare-metal GPU compute (Mike's RTX 4090 with 24GB VRAM) to deliver continuous, high-frame-rate species detection and tracking with $0 marginal cloud inference cost.

---

## 2. Core Architecture

```
[ Public HLS Streams ] (Explore.org / Cornell)
         │
         ▼
[ streamlink / PyAV Ingest ] ──► [ NVIDIA NVDEC (4090) ]
                                          │ (Decoded Frames @ 30-120 FPS)
                                          ▼
                         [ YOLOv11x / MegaDetector v5 ]
                                          │
                         [ Event Dispatcher & SQLite ]
                                          │
                                          ▼
                       [ FastAPI Server-Sent Events (SSE) ]
                                          │
                                          ▼
                 [ Web Ops Room Frontend (Amos UI Lead) ]
```

### A. Video Ingestion & Decoding (Zero Lead)
- **Tooling:** `streamlink` extracting raw HLS chunks (`.m3u8`) from verified public streams.
- **Hardware Acceleration:** Decoded directly via NVIDIA NVDEC into GPU tensor memory, bypassing CPU bottlenecks.
- **Initial Target Feeds:**
  1. Explore.org Anacapa Island Kelp Forest (marine reef / fish).
  2. Cornell Lab FeederWatch (avian identification).
  3. Explore.org Katmai Brooks Falls (brown bears / salmon).

### B. Computer Vision Perception Core (Zero Lead)
- **Model:** YOLOv11x (COCO baseline + fine-tuned wildlife/avian classes) or MegaDetector v5.
- **Inference Mode:** FP16 TensorRT engine running on RTX 4090 CUDA cores (target throughput >120 FPS).
- **Thresholding:** Minimum detection confidence ≥ 0.70 with temporal persistence filter (3 consecutive frames) to eliminate transient false positives.

### C. Event Bus & API (Zero & Amos)
- **Transport:** Server-Sent Events (`GET /events` via FastAPI).
- **Payload Schema:**
  ```json
  {
    "event_id": "evt_10491823",
    "timestamp": "2026-09-21T06:40:00Z",
    "stream_id": "anacapa_kelp_01",
    "species": "garibaldi",
    "confidence": 0.89,
    "bbox": [120, 340, 260, 480],
    "snapshot_url": "/snapshots/20260921/evt_10491823.jpg"
  }
  ```

### D. Web Ops Room Frontend (Amos Lead)
- **Layout:** Dark-theme multi-stream surveillance grid (4–6 concurrent tiles) with dynamic primary viewport switching.
- **Controls:** "Pick-an-animal" filter drawer, recency timeline, and browser audio/visual chime on target species arrival.

---

## 3. Sprint 1 Buildout Plan (48-Hour Cadence)

### Phase 1: Ingest & Perception Pipeline (0–24 Hours)
- [ ] **Task 1.1 (Zero):** Finalize `wildlife_cv_runner.py` with `streamlink` ingest and YOLOv11x CUDA inference loop.
- [ ] **Task 1.2 (Zero):** Implement FastAPI lightweight SSE endpoint (`/events`) and snapshot frame disk buffer.
- [ ] **Task 1.3 (Amos):** Scaffold repository shell (`brockventures/outpost`), setup Next.js/Vite frontend skeleton with EventSource listener.

### Phase 2: UI Integration & Event Dispatch (24–48 Hours)
- [ ] **Task 2.1 (Amos):** Build Ops Room grid UI with auto-highlighting when detection events arrive over SSE.
- [ ] **Task 2.2 (Zero):** Implement temporal confidence filtering and species aggregation to prevent event spam.
- [ ] **Task 2.3 (Zero & Amos):** End-to-end integration test connecting 4090 local runner to frontend web view.

---

## 4. Human UAT & Cadence
- **Cadence:** 48-hour development sprints.
- **Review Drop:** Daily asynchronous standup brief at 7:00 PM PT.
- **Deliverables for Mike & Ryan:**
  - Deployed preview URL (GitHub Pages / Vercel).
  - Test clips with bounding box overlays and detection accuracy metrics.
