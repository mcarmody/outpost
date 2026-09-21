"""Outpost Stream Health Watchdog & Dynamic Failover Resolver.

Monitors live HLS/YouTube stream health, manages token lifecycle, and executes
automatic failover to synthetic frame generation (simulate.py) if an upstream
feed drops, rotates stream IDs, or hits YouTube bot-checks.
"""

import time
from typing import Any, Dict, List, Optional
try:
    import streamlink
    HAS_STREAMLINK = True
except ImportError:
    streamlink = None
    HAS_STREAMLINK = False


class StreamWatchdog:
    def __init__(self, token_ttl_seconds: float = 1800.0):
        self.token_ttl = token_ttl_seconds
        # stream_id -> { "url": str, "resolved_hls": str, "resolved_at": float, "status": str, "consecutive_failures": int }
        self.registry: Dict[str, Dict[str, Any]] = {}

    def register_stream(self, stream_id: str, source_url: str, provider: str = "Explore.org"):
        """Registers a stream target for health monitoring."""
        self.registry[stream_id] = {
            "stream_id": stream_id,
            "source_url": source_url,
            "provider": provider,
            "resolved_hls": None,
            "resolved_at": 0.0,
            "status": "unresolved",
            "consecutive_failures": 0,
            "mode": "live",  # 'live' or 'fallback_synthetic'
            "last_error": None,
        }

    def resolve_stream_url(self, stream_id: str, quality: str = "720p", force: bool = False) -> Dict[str, Any]:
        """Resolves raw HLS URL (.m3u8) using streamlink with caching and synthetic failover."""
        entry = self.registry.get(stream_id)
        if not entry:
            raise ValueError(f"Stream ID {stream_id} not registered in watchdog.")

        now = time.time()
        # Return cached HLS URL if still valid and not forced
        if (
            not force
            and entry["resolved_hls"]
            and entry["status"] == "healthy_live"
            and (now - entry["resolved_at"] < self.token_ttl)
        ):
            return {
                "stream_id": stream_id,
                "url": entry["resolved_hls"],
                "mode": "live",
                "cached": True,
            }

        if not HAS_STREAMLINK:
            entry["consecutive_failures"] += 1
            entry["last_error"] = "streamlink package not installed; running in synthetic fallback mode"
            entry["status"] = "fallback_synthetic"
            entry["mode"] = "fallback_synthetic"
            return {
                "stream_id": stream_id,
                "url": None,
                "mode": "fallback_synthetic",
                "cached": False,
                "fallback_reason": "streamlink package not installed; running in synthetic fallback mode",
            }

        # Attempt streamlink resolution
        source_url = entry["source_url"]
        try:
            session = streamlink.Streamlink()
            session.set_option("hls-live-edge", 3)
            session.set_option("http-timeout", 4.0)

            # Check if mock or synthetic
            if source_url.startswith("mock://") or "synthetic" in source_url:
                raise ValueError("Synthetic stream source")

            streams = session.streams(source_url)
            if not streams:
                raise RuntimeError(f"No active video streams found for {source_url}")

            selected = streams.get(quality) or streams.get("best") or next(iter(streams.values()))
            direct_hls = selected.url

            entry["resolved_hls"] = direct_hls
            entry["resolved_at"] = now
            entry["status"] = "healthy_live"
            entry["consecutive_failures"] = 0
            entry["mode"] = "live"
            entry["last_error"] = None

            return {
                "stream_id": stream_id,
                "url": direct_hls,
                "mode": "live",
                "cached": False,
            }

        except Exception as e:
            entry["consecutive_failures"] += 1
            entry["last_error"] = str(e)
            entry["status"] = "degraded" if entry["consecutive_failures"] < 3 else "fallback_synthetic"
            entry["mode"] = "fallback_synthetic"

            # Smooth fallback to synthetic simulation
            return {
                "stream_id": stream_id,
                "url": None,
                "mode": "fallback_synthetic",
                "cached": False,
                "fallback_reason": str(e),
            }

    def record_frame_result(self, stream_id: str, success: bool):
        """Notifies watchdog of frame read success or drop to drive failover thresholds."""
        entry = self.registry.get(stream_id)
        if not entry:
            return

        if success:
            entry["consecutive_failures"] = 0
            if entry["mode"] == "live":
                entry["status"] = "healthy_live"
        else:
            entry["consecutive_failures"] += 1
            if entry["consecutive_failures"] >= 3:
                entry["mode"] = "fallback_synthetic"
                entry["status"] = "fallback_synthetic"

    def get_watchdog_status(self) -> List[Dict[str, Any]]:
        """Returns watchdog health report across all registered streams."""
        report = []
        for sid, entry in self.registry.items():
            report.append({
                "stream_id": sid,
                "provider": entry["provider"],
                "status": entry["status"],
                "mode": entry["mode"],
                "consecutive_failures": entry["consecutive_failures"],
                "token_age_sec": round(time.time() - entry["resolved_at"], 1) if entry["resolved_at"] else None,
                "last_error": entry["last_error"],
            })
        return report


if __name__ == "__main__":
    wd = StreamWatchdog()
    wd.register_stream("anacapa_kelp_01", "https://www.youtube.com/watch?v=bZ_S8kP_hLg", "Explore.org")
    wd.register_stream("cornell_feeder_01", "mock://unreachable_feed", "Cornell Lab")

    print("[*] Resolving test streams:")
    res1 = wd.resolve_stream_url("cornell_feeder_01")
    print("  Cornell:", res1)
    status = wd.get_watchdog_status()
    print("  Status:", status)
