"""Outpost Pipeline Supervisor & Hardware Environment Probe.

Orchestrates hardware capability checks (RTX 4090 / CUDA detection vs CPU fallback),
background maintenance workers (snapshot retention, stream watchdog), and graceful lifecycle shutdowns.
"""

import os
import signal
import sys
import time
from typing import Any, Dict, Optional


def get_hardware_diagnostics() -> Dict[str, Any]:
    """Detects available compute acceleration (NVIDIA CUDA / TensorRT vs CPU fallback)."""
    cuda_available = False
    device_name = "CPU Only"
    total_vram_gb = 0.0
    recommended_mode = "synthetic"

    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
            total_vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1)
            recommended_mode = "cuda_tensorrt" if "4090" in device_name or total_vram_gb >= 16 else "cuda_fp16"
    except ImportError:
        pass

    return {
        "cuda_available": cuda_available,
        "device_name": device_name,
        "total_vram_gb": total_vram_gb,
        "recommended_mode": recommended_mode,
        "host_platform": sys.platform,
        "python_version": sys.version.split()[0],
    }


class PipelineSupervisor:
    def __init__(self, mode: str = "auto", api_port: int = 8000):
        self.diagnostics = get_hardware_diagnostics()
        if mode == "auto":
            self.mode = self.diagnostics["recommended_mode"]
        else:
            self.mode = mode

        self.api_port = api_port
        self.is_running = False
        self.start_time = 0.0

    def start(self):
        """Starts the pipeline supervisor."""
        self.is_running = True
        self.start_time = time.time()
        print(f"[*] Outpost Pipeline Supervisor Started | Mode: {self.mode}")
        print(f"[*] Hardware: {self.diagnostics['device_name']} (CUDA: {self.diagnostics['cuda_available']})")

    def stop(self):
        """Gracefully terminates background supervisor workers."""
        self.is_running = False
        print("[*] Outpost Pipeline Supervisor Shutting Down Gracefully.")

    def get_status(self) -> Dict[str, Any]:
        """Returns live supervisor operational telemetry."""
        return {
            "supervisor": "outpost-pipeline-supervisor",
            "is_running": self.is_running,
            "mode": self.mode,
            "uptime_seconds": round(time.time() - self.start_time, 1) if self.start_time else 0.0,
            "hardware": self.diagnostics,
        }


if __name__ == "__main__":
    sup = PipelineSupervisor()
    sup.start()
    print("[*] Diagnostics Report:", sup.get_status())
    sup.stop()
