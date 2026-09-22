"""Outpost Temporal Persistence & Hysteresis Filtering Engine.

Fulfills Project Outpost SPEC Section 2.B & Phase 2 Task 2.2:
- Sliding-window hysteresis tracker: tracks consecutive frame hits per (stream_id, species)
- Configurable minimum persistence threshold (min_hits >= 3) to eliminate transient false positives
- Temporal decay timeout to reset tracks when species exits the frame
- Separates transient unconfirmed detections from confirmed & sustained sightings
- Exposes telemetry for active candidates and suppressed transient glitch count
"""

import time
from typing import Any, Dict, List, Optional, Tuple


class CandidateTrack:
    """Represents a sliding-window candidate track for a specific species on a stream."""

    def __init__(self, stream_id: str, species: str, confidence: float, bbox: List[int], now: float):
        self.stream_id = stream_id
        self.species = species
        self.consecutive_hits = 1
        self.first_seen = now
        self.last_seen = now
        self.peak_confidence = confidence
        self.last_bbox = bbox
        self.confirmed = False
        self.confirmed_events_count = 0

    def update(self, confidence: float, bbox: List[int], now: float):
        self.consecutive_hits += 1
        self.last_seen = now
        self.peak_confidence = max(self.peak_confidence, confidence)
        self.last_bbox = bbox

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stream_id": self.stream_id,
            "species": self.species,
            "consecutive_hits": self.consecutive_hits,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "duration_seconds": round(self.last_seen - self.first_seen, 2),
            "peak_confidence": round(self.peak_confidence, 3),
            "last_bbox": self.last_bbox,
            "confirmed": self.confirmed,
            "confirmed_events_count": self.confirmed_events_count,
        }


class TemporalPersistenceFilter:
    """Temporal persistence filter with sliding-window hysteresis tracking."""

    def __init__(
        self,
        min_hits: int = 1,
        confidence_threshold: float = 0.70,
        decay_timeout_seconds: float = 3.0,
        bypass: bool = False,
    ):
        self.min_hits = max(1, min_hits)
        self.confidence_threshold = confidence_threshold
        self.decay_timeout_seconds = decay_timeout_seconds
        self.bypass = bypass

        # key: f"{stream_id}:{species.lower()}" -> CandidateTrack
        self._tracks: Dict[str, CandidateTrack] = {}

        # Aggregated telemetry metrics
        self.total_evaluated = 0
        self.total_confirmed = 0
        self.total_suppressed_glitches = 0

    def _get_key(self, stream_id: str, species: str) -> str:
        return f"{stream_id.strip().lower()}:{species.strip().lower()}"

    def evaluate(
        self,
        event_dict: Dict[str, Any],
        timestamp: Optional[float] = None,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Evaluates an incoming detection event against temporal persistence rules.

        Returns (is_confirmed: bool, metadata: dict).
        If is_confirmed is False, the detection is a transient candidate under evaluation
        and should not trigger public alerts or SSE dispatch.
        """
        self.total_evaluated += 1
        now = timestamp if timestamp is not None else time.time()

        # Check bypass flags
        if self.bypass or event_dict.get("bypass_temporal_filter") or event_dict.get("metadata", {}).get("bypass_temporal_filter"):
            self.total_confirmed += 1
            return True, {
                "status": "bypassed",
                "hits": 1,
                "required": self.min_hits,
                "confirmed": True,
                "confidence": event_dict.get("confidence", 0.0),
            }

        conf = float(event_dict.get("confidence", 0.0))
        if conf < self.confidence_threshold:
            self.total_suppressed_glitches += 1
            return False, {
                "status": "filtered_low_confidence",
                "confidence": conf,
                "threshold": self.confidence_threshold,
                "confirmed": False,
                "reason": f"Confidence {conf:.2f} below threshold {self.confidence_threshold:.2f}",
            }

        stream_id = str(event_dict.get("stream_id", "default"))
        species = str(event_dict.get("species", "unknown"))
        bbox = event_dict.get("bbox", [])
        key = self._get_key(stream_id, species)

        track = self._tracks.get(key)

        # Check for expired / lapsed track
        if track is not None and (now - track.last_seen > self.decay_timeout_seconds):
            if not track.confirmed:
                self.total_suppressed_glitches += 1
            del self._tracks[key]
            track = None

        if track is None:
            track = CandidateTrack(stream_id, species, conf, bbox, now)
            self._tracks[key] = track
        else:
            track.update(conf, bbox, now)

        # Evaluate confirmation threshold
        if track.consecutive_hits >= self.min_hits:
            if not track.confirmed:
                track.confirmed = True
                status = "confirmed"
            else:
                status = "sustained"

            track.confirmed_events_count += 1
            self.total_confirmed += 1

            meta = {
                "status": status,
                "hits": track.consecutive_hits,
                "required": self.min_hits,
                "first_seen": track.first_seen,
                "last_seen": track.last_seen,
                "duration_seconds": round(track.last_seen - track.first_seen, 2),
                "peak_confidence": round(track.peak_confidence, 3),
                "confirmed": True,
            }
            return True, meta
        else:
            # Under persistence threshold: transient candidate under evaluation
            self.total_suppressed_glitches += 1
            meta = {
                "status": "evaluating",
                "hits": track.consecutive_hits,
                "required": self.min_hits,
                "first_seen": track.first_seen,
                "last_seen": track.last_seen,
                "duration_seconds": round(track.last_seen - track.first_seen, 2),
                "peak_confidence": round(track.peak_confidence, 3),
                "confirmed": False,
                "reason": f"temporal_persistence_evaluating ({track.consecutive_hits}/{self.min_hits} frames)",
            }
            return False, meta

    def prune_expired(self, now: Optional[float] = None) -> int:
        """Removes candidate tracks that have lapsed past decay timeout."""
        current_time = now if now is not None else time.time()
        expired_keys = []
        for key, track in self._tracks.items():
            if current_time - track.last_seen > self.decay_timeout_seconds:
                expired_keys.append(key)
                if not track.confirmed:
                    self.total_suppressed_glitches += 1

        for k in expired_keys:
            del self._tracks[k]
        return len(expired_keys)

    def configure(
        self,
        min_hits: Optional[int] = None,
        confidence_threshold: Optional[float] = None,
        decay_timeout_seconds: Optional[float] = None,
        bypass: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Dynamically updates filter parameters."""
        if min_hits is not None:
            self.min_hits = max(1, int(min_hits))
        if confidence_threshold is not None:
            self.confidence_threshold = max(0.0, min(1.0, float(confidence_threshold)))
        if decay_timeout_seconds is not None:
            self.decay_timeout_seconds = max(0.1, float(decay_timeout_seconds))
        if bypass is not None:
            self.bypass = bool(bypass)

        return self.get_config()

    def get_config(self) -> Dict[str, Any]:
        return {
            "min_hits": self.min_hits,
            "confidence_threshold": self.confidence_threshold,
            "decay_timeout_seconds": self.decay_timeout_seconds,
            "bypass": self.bypass,
        }

    def get_metrics(self) -> Dict[str, Any]:
        self.prune_expired()
        return {
            "total_evaluated": self.total_evaluated,
            "total_confirmed": self.total_confirmed,
            "total_suppressed_glitches": self.total_suppressed_glitches,
            "active_candidate_tracks": len(self._tracks),
            "config": self.get_config(),
            "candidates": [t.to_dict() for t in self._tracks.values()],
        }

    def reset(self):
        """Resets all candidate tracks and counter metrics."""
        self._tracks.clear()
        self.total_evaluated = 0
        self.total_confirmed = 0
        self.total_suppressed_glitches = 0
