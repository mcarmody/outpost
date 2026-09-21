"""Wildlife & Public Stream CV Ingest Runner (Optimized for RTX 4090 / CUDA)

Prerequisites on Mike's local GPU box:
  pip install streamlink opencv-python ultralytics torch torchvision requests

Usage:
  python runner.py --url "https://www.youtube.com/watch?v=..." --model yolo11x.pt --api-url "http://localhost:8000"
"""

import argparse
import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

import cv2
import requests
import streamlink
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
SNAPSHOTS_DIR = BASE_DIR / "snapshots"
SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)


def get_stream_url(youtube_url: str, quality: str = "1080p") -> str:
    session = streamlink.Streamlink()
    session.set_option("hls-live-edge", 3)
    streams = session.streams(youtube_url)
    if not streams:
        raise ValueError(f"No streams found for {youtube_url}")
    if quality in streams:
        return streams[quality].url
    return streams["best"].url


def dispatch_event(
    api_url: Optional[str],
    stream_id: str,
    species: str,
    confidence: float,
    bbox: list,
    frame=None,
):
    event_id = f"evt_{uuid.uuid4().hex[:8]}"
    today_str = time.strftime("%Y%m%d")
    snapshot_rel_path = f"/snapshots/{today_str}/{event_id}.jpg"

    if frame is not None:
        try:
            day_dir = SNAPSHOTS_DIR / today_str
            day_dir.mkdir(parents=True, exist_ok=True)
            img_path = day_dir / f"{event_id}.jpg"
            cv2.imwrite(str(img_path), frame)
        except Exception:
            pass

    payload = {
        "event_id": event_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stream_id": stream_id,
        "species": species,
        "confidence": round(confidence, 3),
        "bbox": bbox,
        "snapshot_url": snapshot_rel_path,
    }

    print(f"[{time.strftime('%X')}] DETECTED: {species} ({confidence:.2f}) -> {event_id}")

    if api_url:
        base = api_url.rstrip("/")
        try:
            if frame is not None:
                # Multipart upload of JPEG frame and event metadata
                ok, img_buf = cv2.imencode(".jpg", frame)
                if ok:
                    files = {"file": (f"{event_id}.jpg", img_buf.tobytes(), "image/jpeg")}
                    data = {"event_data": json.dumps(payload)}
                    res = requests.post(f"{base}/api/events/upload", files=files, data=data, timeout=3.0)
                    if res.status_code in (200, 201):
                        r_data = res.json()
                        if r_data.get("filtered"):
                            print(f"  [i] Stream allowlist filtered: {r_data.get('reason')}")
                        return
            # Fallback to standard JSON dispatch
            res = requests.post(f"{base}/api/events", json=payload, timeout=2.0)
            if res.status_code in (200, 201) and res.json().get("filtered"):
                print(f"  [i] Stream allowlist filtered: {res.json().get('reason')}")
        except Exception as e:
            print(f"[!] Warning: Failed to dispatch event to {api_url}: {e}")


def _hardware_payload(stream_id: str) -> dict:
    import platform
    import socket
    import torch
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU Only"
    total_vram_gb = (
        round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1)
        if cuda_available else 0.0
    )
    return {
        "stream_id": stream_id,
        "cuda_available": cuda_available,
        "device_name": device_name,
        "total_vram_gb": total_vram_gb,
        "hostname": f"{socket.gethostname()} ({platform.system()})",
    }


def register_hardware_once(api_url: Optional[str], stream_id: str) -> bool:
    """Single hardware registration attempt. Returns True on confirmed success."""
    if not api_url:
        return False
    try:
        payload = _hardware_payload(stream_id)
        res = requests.post(f"{api_url.rstrip('/')}/api/runner/register", json=payload, timeout=3.0)
        if res.status_code in (200, 201):
            print(f"[*] Registered hardware with {api_url}: {payload['device_name']}")
            return True
        print(f"[!] Warning: hardware registration got HTTP {res.status_code} "
              f"(server may not have the /api/runner/register route deployed yet)")
        return False
    except Exception as e:
        print(f"[!] Warning: hardware registration failed: {e}")
        return False


def register_hardware_loop(api_url: Optional[str], stream_id: str, interval_seconds: float = 60.0):
    """Re-registers on a fixed interval for the life of the process, run in
    a background thread. The relay's registration is in-memory with a TTL
    (server.py: EDGE_HARDWARE_TTL_SECONDS) — a relay restart or redeploy
    (the Host 2 auto-deploy watcher bounces Uvicorn on every push to main)
    silently wipes it, and a once-at-startup registration then never comes
    back until someone notices the dashboard says 'no edge GPU runner' and
    manually restarts this process. Re-registering periodically means the
    next relay restart just gets picked back up within one interval,
    automatically. Found live 2026-09-21 after exactly that happened twice
    in a row following auto-deploy restarts."""
    while True:
        register_hardware_once(api_url, stream_id)
        time.sleep(interval_seconds)


def run_pipeline(stream_url: str, stream_id: str = "anacapa_kelp_01", model_name: str = "yolo11x.pt", api_url: Optional[str] = "http://localhost:8000"):
    threading.Thread(
        target=register_hardware_loop, args=(api_url, stream_id), daemon=True
    ).start()

    print(f"[*] Loading model {model_name} onto CUDA...")
    model = YOLO(model_name)
    model.to("cuda")

    print(f"[*] Opening stream: {stream_url[:60]}...")
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        raise RuntimeError("Failed to open video stream")

    frame_count = 0
    t0 = time.time()

    # Track consecutive detections for temporal persistence filter
    detection_history = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[!] Frame drop or stream disconnected, reconnecting...")
            time.sleep(1)
            continue

        frame_count += 1
        # Inference on CUDA (TensorRT or FP16)
        results = model.predict(frame, device=0, half=True, verbose=False)

        # Process detections
        current_frame_species = set()
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names[cls_id]
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]

                if conf >= 0.70:
                    current_frame_species.add(label)
                    detection_history[label] = detection_history.get(label, 0) + 1

                    # Temporal persistence: dispatch when detected across 3 consecutive frames
                    if detection_history[label] == 3:
                        dispatch_event(
                            api_url=api_url,
                            stream_id=stream_id,
                            species=label,
                            confidence=conf,
                            bbox=[x1, y1, x2, y2],
                            frame=frame,
                        )

        # Decay history for species not in current frame
        for sp in list(detection_history.keys()):
            if sp not in current_frame_species:
                detection_history[sp] = max(0, detection_history[sp] - 1)

        if frame_count % 100 == 0:
            fps = frame_count / (time.time() - t0)
            print(f"[*] Processed {frame_count} frames | Avg FPS: {fps:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default="https://www.youtube.com/watch?v=OAJF1Ie1m_Q", help="YouTube/Explore stream URL")
    parser.add_argument("--stream-id", type=str, default="anacapa_kelp_01", help="Identifier for stream")
    parser.add_argument("--model", type=str, default="yolo11x.pt", help="YOLO model or checkpoint")
    parser.add_argument("--api-url", type=str, default="http://localhost:8000", help="Outpost FastAPI telemetry server URL")
    args = parser.parse_args()

    direct_url = get_stream_url(args.url)
    run_pipeline(direct_url, args.stream_id, args.model, args.api_url)
