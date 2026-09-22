"""Outpost Wildlife Sighting Session & Encounter Tracker.

Clusters high-frequency raw YOLO frame detections into coherent wildlife sighting
sessions / encounters:
- Groups detections by (stream_id, species).
- Detections occurring within `gap_threshold_seconds` (default 60s) are merged into
  the active sighting session.
- Tracks session start, end, duration (dwell time), detection count, peak confidence,
  and the highest-confidence snapshot URL.
- When an animal departs (no detections for >gap_threshold_seconds), the session
  transitions to 'completed'.
- Provides session analytics: average dwell time per species, longest encounters,
  and active visitors across cameras.
"""

import time
import uuid
from typing import Any, Dict, List, Optional, Tuple


class SightingSessionTracker:
    def __init__(self, gap_threshold_seconds: float = 60.0, max_completed_history: int = 200):
        self.gap_threshold = float(gap_threshold_seconds)
        # Key: "stream_id:species_norm" -> Session dict
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.completed_sessions: List[Dict[str, Any]] = []
        self.max_completed_history = max_completed_history

    def _session_key(self, stream_id: str, species: str) -> str:
        s_norm = (stream_id or "").strip().lower()
        sp_norm = (species or "").strip().lower().replace(" ", "_")
        return f"{s_norm}:{sp_norm}"

    def process_event(
        self,
        event: Dict[str, Any],
        now_ts: Optional[float] = None,
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """Ingests a detection event and clusters it into an active or new sighting session.

        Returns:
            Tuple of (active_session, newly_completed_session_if_any)
        """
        now = now_ts if now_ts is not None else time.time()
        stream_id = event.get("stream_id", "default")
        species = event.get("species", "unknown")
        confidence = float(event.get("confidence", 0.0))
        snapshot_url = event.get("snapshot_url")
        timestamp_str = event.get("timestamp") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))

        key = self._session_key(stream_id, species)
        completed_session = None

        if key in self.active_sessions:
            sess = self.active_sessions[key]
            time_since_last = now - sess["last_seen_ts"]

            if time_since_last <= self.gap_threshold:
                # Continuation of existing active encounter
                sess["detection_count"] += 1
                sess["end_time"] = timestamp_str
                sess["last_seen_ts"] = now
                sess["duration_seconds"] = round(now - sess["start_ts"], 1)

                if confidence >= sess["peak_confidence"]:
                    sess["peak_confidence"] = confidence
                    if snapshot_url:
                        sess["best_snapshot_url"] = snapshot_url

                return sess, None
            else:
                # Prior session timed out; complete it and rotate
                sess["status"] = "completed"
                completed_session = dict(sess)
                self.completed_sessions.append(completed_session)
                if len(self.completed_sessions) > self.max_completed_history:
                    self.completed_sessions.pop(0)

        # Initialize new active sighting encounter
        new_sess = {
            "session_id": f"sess_{uuid.uuid4().hex[:12]}",
            "stream_id": stream_id,
            "species": species,
            "start_time": timestamp_str,
            "end_time": timestamp_str,
            "start_ts": now,
            "last_seen_ts": now,
            "duration_seconds": 0.0,
            "detection_count": 1,
            "peak_confidence": confidence,
            "best_snapshot_url": snapshot_url,
            "status": "active",
        }
        self.active_sessions[key] = new_sess
        return new_sess, completed_session

    def sweep_expired_sessions(self, now_ts: Optional[float] = None) -> List[Dict[str, Any]]:
        """Sweeps and completes any active sessions that have had no detections beyond gap_threshold."""
        now = now_ts if now_ts is not None else time.time()
        expired_keys = []
        newly_completed = []

        for key, sess in list(self.active_sessions.items()):
            if (now - sess["last_seen_ts"]) > self.gap_threshold:
                sess["status"] = "completed"
                completed_dict = dict(sess)
                newly_completed.append(completed_dict)
                self.completed_sessions.append(completed_dict)
                if len(self.completed_sessions) > self.max_completed_history:
                    self.completed_sessions.pop(0)
                expired_keys.append(key)

        for k in expired_keys:
            del self.active_sessions[k]

        return newly_completed

    def get_active_sessions(self, now_ts: Optional[float] = None) -> List[Dict[str, Any]]:
        """Returns all currently active wildlife encounters."""
        self.sweep_expired_sessions(now_ts=now_ts)
        return list(self.active_sessions.values())

    def get_recent_sessions(
        self,
        limit: int = 50,
        stream_id: Optional[str] = None,
        species: Optional[str] = None,
        status: Optional[str] = None,
        include_active: bool = True,
        now_ts: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Returns recent wildlife sessions filtered by stream, species, or status."""
        self.sweep_expired_sessions(now_ts=now_ts)
        all_sessions = []
        if include_active:
            all_sessions.extend(self.active_sessions.values())
        all_sessions.extend(self.completed_sessions)

        filtered = []
        for s in all_sessions:
            if stream_id and s.get("stream_id") != stream_id:
                continue
            if species and s.get("species", "").lower() != species.lower():
                continue
            if status and s.get("status") != status:
                continue
            filtered.append(s)

        # Sort descending by start_time
        filtered.sort(key=lambda x: x.get("start_time", ""), reverse=True)
        return filtered[:limit]

    def get_session_analytics(self) -> Dict[str, Any]:
        """Calculates dwell time metrics, visit counts, and species engagement."""
        self.sweep_expired_sessions()
        all_completed = list(self.completed_sessions)
        total_completed = len(all_completed)

        dwell_times = [s["duration_seconds"] for s in all_completed]
        avg_dwell = round(sum(dwell_times) / total_completed, 1) if total_completed > 0 else 0.0
        max_sess = max(all_completed, key=lambda s: s["duration_seconds"]) if all_completed else None

        species_agg: Dict[str, Dict[str, Any]] = {}
        stream_agg: Dict[str, int] = {}

        for s in all_completed:
            sp = s.get("species", "unknown")
            sid = s.get("stream_id", "default")
            stream_agg[sid] = stream_agg.get(sid, 0) + 1

            if sp not in species_agg:
                species_agg[sp] = {
                    "species": sp,
                    "visit_count": 0,
                    "total_dwell_seconds": 0.0,
                    "max_dwell_seconds": 0.0,
                    "peak_confidence": 0.0,
                    "total_detections": 0,
                }
            species_agg[sp]["visit_count"] += 1
            species_agg[sp]["total_dwell_seconds"] += s["duration_seconds"]
            species_agg[sp]["max_dwell_seconds"] = max(species_agg[sp]["max_dwell_seconds"], s["duration_seconds"])
            species_agg[sp]["peak_confidence"] = max(species_agg[sp]["peak_confidence"], s["peak_confidence"])
            species_agg[sp]["total_detections"] += s.get("detection_count", 1)

        species_metrics = []
        for sp, stats in species_agg.items():
            species_metrics.append({
                "species": sp,
                "visit_count": stats["visit_count"],
                "avg_dwell_seconds": round(stats["total_dwell_seconds"] / stats["visit_count"], 1),
                "max_dwell_seconds": stats["max_dwell_seconds"],
                "peak_confidence": stats["peak_confidence"],
                "total_detections": stats["total_detections"],
            })
        species_metrics.sort(key=lambda x: (x["visit_count"], x["avg_dwell_seconds"]), reverse=True)

        return {
            "total_completed_sessions": total_completed,
            "active_sessions_count": len(self.active_sessions),
            "average_dwell_seconds": avg_dwell,
            "longest_encounter": max_sess,
            "stream_session_counts": stream_agg,
            "species_encounter_metrics": species_metrics,
        }
