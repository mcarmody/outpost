"""Wildlife & Public Stream CV Ingest Runner (Optimized for RTX 4090 / CUDA)

Prerequisites on Mike's local GPU box:
  pip install streamlink opencv-python ultralytics torch torchvision

Usage:
  python wildlife_cv_runner.py --url "https://www.youtube.com/watch?v=..." --model yolo11x.pt
"""

import argparse
import time
import cv2
import streamlink
from ultralytics import YOLO


def get_stream_url(youtube_url: str, quality: str = "1080p") -> str:
    session = streamlink.Streamlink()
    session.set_option("hls-live-edge", 3)
    streams = session.streams(youtube_url)
    if not streams:
        raise ValueError(f"No streams found for {youtube_url}")
    if quality in streams:
        return streams[quality].url
    return streams["best"].url


def run_pipeline(stream_url: str, model_name: str = "yolo11x.pt"):
    print(f"[*] Loading model {model_name} onto CUDA...")
    model = YOLO(model_name)
    model.to("cuda")

    print(f"[*] Opening stream: {stream_url[:60]}...")
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        raise RuntimeError("Failed to open video stream")

    frame_count = 0
    t0 = time.time()

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
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names[cls_id]
                if conf >= 0.70:
                    # Log detection event
                    print(f"[{time.strftime('%X')}] DETECTED: {label} ({conf:.2f})")

        if frame_count % 100 == 0:
            fps = frame_count / (time.time() - t0)
            print(f"[*] Processed {frame_count} frames | Avg FPS: {fps:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default="https://www.youtube.com/watch?v=bZ_S8kP_hLg", help="YouTube/Explore stream URL")
    parser.add_argument("--model", type=str, default="yolo11x.pt", help="YOLO model or checkpoint")
    args = parser.parse_args()

    direct_url = get_stream_url(args.url)
    run_pipeline(direct_url, args.model)
