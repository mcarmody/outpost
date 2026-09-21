"""Synthetic Stream & Detection Event Simulator for Project Outpost.

Generates realistic wildlife/marine detection events and renders snapshot
frames so the web frontend can be developed and QA'd without requiring
an active NVIDIA RTX 4090 GPU or live streamlink capture session.
"""

import argparse
import json
import math
import os
import random
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from PIL import Image, ImageDraw

SNAPSHOTS_DIR = Path("/workspace/scratch/outpost/snapshots")
SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)

# Simulated stream configurations
STREAMS = {
    "anacapa_kelp_01": {
        "name": "Explore.org Anacapa Kelp Forest",
        "bg_color": (10, 35, 45),
        "species": [
            ("Garibaldi", (255, 140, 0)),
            ("Giant Kelp Bass", (120, 160, 140)),
            ("California Sea Lion", (180, 150, 120)),
            ("Bat Ray", (80, 100, 120)),
            ("Senorita Fish", (220, 180, 100)),
        ],
    },
    "cornell_feeder_01": {
        "name": "Cornell Lab FeederWatch",
        "bg_color": (25, 30, 25),
        "species": [
            ("Northern Cardinal", (220, 40, 40)),
            ("Blue Jay", (60, 130, 240)),
            ("Black-capped Chickadee", (190, 190, 190)),
            ("Mourning Dove", (160, 140, 130)),
            ("Tufted Titmouse", (140, 150, 170)),
        ],
    },
    "katmai_brooks_falls": {
        "name": "Katmai Brooks Falls Brown Bears",
        "bg_color": (20, 30, 35),
        "species": [
            ("Brown Bear", (139, 90, 43)),
            ("Sockeye Salmon", (230, 80, 70)),
            ("Bald Eagle", (240, 230, 210)),
            ("Glaucous-winged Gull", (210, 215, 220)),
        ],
    },
}


def generate_synthetic_snapshot(
    event_id: str,
    stream_id: str,
    species: str,
    confidence: float,
    bbox: List[int],
    bg_color: Tuple[int, int, int],
    tag_color: Tuple[int, int, int],
) -> str:
    """Renders a synthetic frame image with bounding box and HUD overlays."""
    width, height = 640, 360
    img = Image.new("RGB", (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)

    # Subtle ambient gradient/noise
    for y in range(0, height, 4):
        shade = int(10 * math.sin(y / 20.0))
        draw.line(
            [(0, y), (width, y)],
            fill=(max(0, bg_color[0] + shade), max(0, bg_color[1] + shade), max(0, bg_color[2] + shade)),
        )

    x1, y1, x2, y2 = bbox
    # Draw bounding box
    draw.rectangle([x1, y1, x2, y2], outline=tag_color, width=2)

    # Draw label badge
    badge_text = f"{species} ({int(confidence * 100)}%)"
    text_w = len(badge_text) * 7
    draw.rectangle([x1, max(0, y1 - 18), x1 + text_w + 8, y1], fill=tag_color)
    draw.text((x1 + 4, max(0, y1 - 16)), badge_text, fill=(255, 255, 255))

    # Draw camera HUD overlay (top left)
    hud_text = f"CAM: {stream_id} | {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}"
    draw.rectangle([8, 8, 8 + len(hud_text) * 7 + 8, 26], fill=(0, 0, 0, 160))
    draw.text((12, 11), hud_text, fill=(180, 220, 240))

    # Save snapshot
    today_str = time.strftime("%Y%m%d")
    day_dir = SNAPSHOTS_DIR / today_str
    day_dir.mkdir(parents=True, exist_ok=True)
    file_path = day_dir / f"{event_id}.jpg"
    img.save(file_path, format="JPEG", quality=85)

    return f"/snapshots/{today_str}/{event_id}.jpg"


def simulate_event(api_url: Optional[str] = "http://localhost:8000") -> Dict:
    """Generates and dispatches a single realistic detection event."""
    stream_id = random.choice(list(STREAMS.keys()))
    cfg = STREAMS[stream_id]
    species_name, color = random.choice(cfg["species"])

    event_id = f"evt_{uuid.uuid4().hex[:8]}"
    conf = round(random.uniform(0.74, 0.98), 3)

    # Generate plausible bounding box in 640x360 coordinates
    box_w = random.randint(60, 160)
    box_h = random.randint(40, 120)
    x1 = random.randint(40, 640 - box_w - 40)
    y1 = random.randint(40, 360 - box_h - 40)
    bbox = [x1, y1, x1 + box_w, y1 + box_h]

    # Generate synthetic snapshot image
    snapshot_url = generate_synthetic_snapshot(
        event_id=event_id,
        stream_id=stream_id,
        species=species_name,
        confidence=conf,
        bbox=bbox,
        bg_color=cfg["bg_color"],
        tag_color=color,
    )

    payload = {
        "event_id": event_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stream_id": stream_id,
        "species": species_name,
        "confidence": conf,
        "bbox": bbox,
        "snapshot_url": snapshot_url,
        "metadata": {
            "camera_name": cfg["name"],
            "simulated": True,
            "weather": "clear",
        },
    }

    if api_url:
        try:
            resp = requests.post(f"{api_url.rstrip('/')}/api/events", json=payload, timeout=2.0)
            if resp.status_code == 201:
                print(f"[Sim] Dispatched: {species_name} ({conf:.2f}) on {stream_id} -> {event_id}")
            else:
                print(f"[Sim] Warning: API returned HTTP {resp.status_code}")
        except Exception as e:
            print(f"[Sim] Notice: Server not reachable at {api_url} ({e})")

    return payload


def main():
    parser = argparse.ArgumentParser(description="Outpost Synthetic Event Stream Generator")
    parser.add_argument("--api-url", type=str, default="http://localhost:8000", help="Outpost server endpoint")
    parser.add_argument("--count", type=int, default=10, help="Number of events to generate (0 for continuous)")
    parser.add_argument("--interval", type=float, default=2.0, help="Seconds between events")
    args = parser.parse_args()

    print(f"[*] Starting Outpost synthetic feed generator (target: {args.api_url})...")

    if args.count > 0:
        for i in range(args.count):
            simulate_event(args.api_url)
            if i < args.count - 1 and args.interval > 0:
                time.sleep(args.interval)
    else:
        print("[*] Running continuous simulation (Ctrl+C to stop)...")
        try:
            while True:
                simulate_event(args.api_url)
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n[*] Simulator stopped.")


if __name__ == "__main__":
    main()
