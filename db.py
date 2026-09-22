"""SQLite Persistent Archival and Query Engine for Project Outpost.

Provides durable storage, indexing, and restart survivability for detection events:
- Schema: detection_events table with composite indexes on stream_id, species, and timestamp
- WAL mode and busy timeout for concurrent safety
- Dual-write integration with in-memory ring buffer
- Fast indexed querying for /events/recent, /api/events/{event_id}, and telemetry stats
- Survivability across server restarts and container autodeploys
"""

import json
import logging
import os
import sqlite3
import time
import uuid
from pathlib import Path

logger = logging.getLogger("outpost.db")
from typing import Any, Dict, List, Optional, Union

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = BASE_DIR / "events.db"


def get_db_path(db_path: Optional[Union[Path, str]] = None) -> Path:
    """Resolve the active database path from argument or environment."""
    if db_path is not None:
        return Path(db_path)
    env_path = os.environ.get("OUTPOST_DB_PATH")
    if env_path:
        return Path(env_path)
    return DEFAULT_DB_PATH


def get_db_connection(db_path: Optional[Union[Path, str]] = None) -> sqlite3.Connection:
    """Create and configure a SQLite connection with row factory and busy timeouts."""
    resolved_path = get_db_path(db_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(resolved_path), timeout=5.0)
    conn.row_factory = sqlite3.Row
    if "/tmp" in str(resolved_path) or "test" in str(resolved_path):
        conn.execute("PRAGMA synchronous = OFF;")
    else:
        conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute("PRAGMA busy_timeout = 5000;")
    return conn


def init_db(db_path: Optional[Union[Path, str]] = None) -> None:
    """Initialize SQLite tables, WAL journal mode, and indexes for Outpost detection events."""
    conn = get_db_connection(db_path)
    try:
        conn.execute("PRAGMA journal_mode = WAL;")
        with conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS detection_events (
                    event_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    stream_id TEXT NOT NULL,
                    species TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    bbox_json TEXT NOT NULL,
                    snapshot_url TEXT,
                    metadata_json TEXT
                );
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_events_stream_timestamp
                ON detection_events (stream_id, timestamp DESC);
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_events_species_timestamp
                ON detection_events (species, timestamp DESC);
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_events_timestamp
                ON detection_events (timestamp DESC);
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sighting_sessions (
                    session_id TEXT PRIMARY KEY,
                    stream_id TEXT NOT NULL,
                    species TEXT NOT NULL,
                    start_time TEXT NOT NULL,
                    end_time TEXT NOT NULL,
                    duration_seconds REAL NOT NULL DEFAULT 0.0,
                    detection_count INTEGER NOT NULL DEFAULT 1,
                    peak_confidence REAL NOT NULL,
                    best_snapshot_url TEXT,
                    status TEXT NOT NULL DEFAULT 'active'
                );
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_sessions_stream_start
                ON sighting_sessions (stream_id, start_time DESC);
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_sessions_species_start
                ON sighting_sessions (species, start_time DESC);
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_sessions_status
                ON sighting_sessions (status);
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS human_reviews (
                    review_id TEXT PRIMARY KEY,
                    event_id TEXT NOT NULL,
                    stream_id TEXT NOT NULL,
                    proposed_species TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    snapshot_url TEXT,
                    bbox_json TEXT,
                    decision TEXT NOT NULL,
                    confirmed_species TEXT,
                    notes TEXT,
                    reviewer TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_reviews_event
                ON human_reviews (event_id);
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_reviews_stream_time
                ON human_reviews (stream_id, timestamp DESC);
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_reviews_decision
                ON human_reviews (decision);
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_reviews_reviewer
                ON human_reviews (reviewer);
                """
            )
    finally:
        conn.close()


def save_event(
    event: Any,
    db_path: Optional[Union[Path, str]] = None,
) -> bool:
    """Persist a detection event (Pydantic model or dict) to SQLite."""
    if hasattr(event, "model_dump"):
        data = event.model_dump()
    elif isinstance(event, dict):
        data = event
    else:
        raise ValueError(f"Unsupported event type: {type(event)}")

    event_id = data.get("event_id")
    timestamp = data.get("timestamp") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    stream_id = data.get("stream_id", "")
    species = data.get("species", "")
    confidence = float(data.get("confidence", 0.0))
    bbox = data.get("bbox", [])
    snapshot_url = data.get("snapshot_url")
    metadata = data.get("metadata", {})

    bbox_json = json.dumps(bbox)
    metadata_json = json.dumps(metadata)

    conn = get_db_connection(db_path)
    try:
        with conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO detection_events (
                    event_id, timestamp, stream_id, species, confidence, bbox_json, snapshot_url, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (event_id, timestamp, stream_id, species, confidence, bbox_json, snapshot_url, metadata_json),
            )
        return True
    except Exception as e:
        print(f"[db] error saving event {event_id}: {e}")
        return False
    finally:
        conn.close()


def row_to_event_dict(row: sqlite3.Row) -> Dict[str, Any]:
    """Convert SQLite row to DetectionEvent dictionary format."""
    try:
        bbox = json.loads(row["bbox_json"]) if row["bbox_json"] else []
    except Exception:
        bbox = []

    try:
        metadata = json.loads(row["metadata_json"]) if row["metadata_json"] else {}
    except Exception:
        metadata = {}

    return {
        "event_id": row["event_id"],
        "timestamp": row["timestamp"],
        "stream_id": row["stream_id"],
        "species": row["species"],
        "confidence": float(row["confidence"]),
        "bbox": bbox,
        "snapshot_url": row["snapshot_url"],
        "metadata": metadata,
    }


def get_event(
    event_id: str,
    db_path: Optional[Union[Path, str]] = None,
) -> Optional[Dict[str, Any]]:
    """Retrieve a specific detection event by ID."""
    conn = get_db_connection(db_path)
    try:
        cursor = conn.execute(
            """
            SELECT event_id, timestamp, stream_id, species, confidence, bbox_json, snapshot_url, metadata_json
            FROM detection_events
            WHERE event_id = ?
            """,
            (event_id,),
        )
        row = cursor.fetchone()
        if row:
            return row_to_event_dict(row)
        return None
    finally:
        conn.close()


def query_events(
    limit: int = 50,
    stream_id: Optional[str] = None,
    species: Optional[str] = None,
    since: Optional[str] = None,
    order: str = "asc",
    db_path: Optional[Union[Path, str]] = None,
) -> List[Dict[str, Any]]:
    """Query detection events with indexed filtering and ordering.

    When order='asc' (default for ring buffer parity), queries the most recent `limit`
    events and returns them in chronological order (oldest to newest).
    When order='desc', returns newest first.
    """
    conn = get_db_connection(db_path)
    try:
        where_clauses = []
        params: List[Any] = []

        if stream_id:
            where_clauses.append("stream_id = ?")
            params.append(stream_id)

        if species:
            sp_norm = species.strip().lower().replace("_", " ")
            where_clauses.append("LOWER(REPLACE(species, '_', ' ')) = ?")
            params.append(sp_norm)

        if since:
            where_clauses.append("timestamp >= ?")
            params.append(since)

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        query_sql = f"""
            SELECT event_id, timestamp, stream_id, species, confidence, bbox_json, snapshot_url, metadata_json
            FROM detection_events
            {where_sql}
            ORDER BY timestamp DESC, rowid DESC
            LIMIT ?
        """
        params.append(max(1, limit))

        cursor = conn.execute(query_sql, params)
        rows = cursor.fetchall()
        events = [row_to_event_dict(r) for r in rows]

        if order == "asc":
            events.reverse()
        return events
    finally:
        conn.close()


def get_species_stats_summary(
    stream_id: Optional[str] = None,
    db_path: Optional[Union[Path, str]] = None,
) -> List[Dict[str, Any]]:
    """Retrieve aggregated species stats directly from persistent SQLite storage."""
    conn = get_db_connection(db_path)
    try:
        if stream_id:
            cursor = conn.execute(
                """
                SELECT species, COUNT(*) as count, MAX(confidence) as peak_confidence,
                       MAX(timestamp) as last_seen, GROUP_CONCAT(DISTINCT stream_id) as streams_csv
                FROM detection_events
                WHERE stream_id = ?
                GROUP BY species
                ORDER BY count DESC
                """,
                (stream_id,),
            )
        else:
            cursor = conn.execute(
                """
                SELECT species, COUNT(*) as count, MAX(confidence) as peak_confidence,
                       MAX(timestamp) as last_seen, GROUP_CONCAT(DISTINCT stream_id) as streams_csv
                FROM detection_events
                GROUP BY species
                ORDER BY count DESC
                """
            )
        rows = cursor.fetchall()
        stats = []
        for r in rows:
            streams = [s.strip() for s in r["streams_csv"].split(",") if s.strip()] if r["streams_csv"] else []
            stats.append({
                "species": r["species"],
                "count": r["count"],
                "peak_confidence": round(float(r["peak_confidence"]), 2),
                "last_seen": r["last_seen"],
                "streams": streams,
            })
        return stats
    finally:
        conn.close()


def get_hourly_activity(
    hours: int = 24,
    stream_id: Optional[str] = None,
    db_path: Optional[Union[Path, str]] = None,
) -> Dict[str, Any]:
    """Retrieve time-bucketed hourly activity trends, peak hour, and species distribution."""
    hours = max(1, min(hours, 168))
    cutoff_time = time.time() - (hours * 3600.0)
    cutoff_str = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(cutoff_time))

    conn = get_db_connection(db_path)
    try:
        where_clauses = ["timestamp >= ?"]
        params: List[Any] = [cutoff_str]

        if stream_id:
            where_clauses.append("stream_id = ?")
            params.append(stream_id)

        where_sql = f"WHERE {' AND '.join(where_clauses)}"

        # Query bucketed hourly stats
        query_buckets = f"""
            SELECT substr(timestamp, 1, 13) || ':00:00Z' AS hour_bucket,
                   COUNT(*) AS count,
                   COUNT(DISTINCT species) AS unique_species,
                   GROUP_CONCAT(DISTINCT stream_id) AS streams_csv
            FROM detection_events
            {where_sql}
            GROUP BY hour_bucket
            ORDER BY hour_bucket ASC
        """
        cursor = conn.execute(query_buckets, params)
        rows = cursor.fetchall()

        # Query top species per hour
        query_top_species = f"""
            SELECT substr(timestamp, 1, 13) || ':00:00Z' AS hour_bucket,
                   species,
                   COUNT(*) AS sp_count
            FROM detection_events
            {where_sql}
            GROUP BY hour_bucket, species
            ORDER BY hour_bucket ASC, sp_count DESC
        """
        cursor_sp = conn.execute(query_top_species, params)
        sp_rows = cursor_sp.fetchall()

        hour_top_sp: Dict[str, str] = {}
        for r in sp_rows:
            hb = r["hour_bucket"]
            if hb not in hour_top_sp:
                hour_top_sp[hb] = r["species"]

        buckets = []
        total_detections = 0
        peak_hour = None
        peak_count = 0
        all_streams = set()

        for r in rows:
            hb = r["hour_bucket"]
            cnt = r["count"]
            total_detections += cnt
            streams = [s.strip() for s in r["streams_csv"].split(",") if s.strip()] if r["streams_csv"] else []
            all_streams.update(streams)

            if cnt > peak_count:
                peak_count = cnt
                peak_hour = hb

            buckets.append({
                "hour": hb,
                "count": cnt,
                "unique_species": r["unique_species"],
                "top_species": hour_top_sp.get(hb, "unknown"),
                "streams": streams,
            })

        # Species distribution within window
        query_dist = f"""
            SELECT species, COUNT(*) as count
            FROM detection_events
            {where_sql}
            GROUP BY species
            ORDER BY count DESC
        """
        cursor_dist = conn.execute(query_dist, params)
        species_dist = [{"species": r["species"], "count": r["count"]} for r in cursor_dist.fetchall()]

        return {
            "time_window_hours": hours,
            "since": cutoff_str,
            "total_detections": total_detections,
            "unique_species": len(species_dist),
            "peak_hour": {"hour": peak_hour, "count": peak_count} if peak_hour else None,
            "hourly_buckets": buckets,
            "species_distribution": species_dist,
            "active_streams": sorted(all_streams),
        }
    finally:
        conn.close()


def get_db_stats(db_path: Optional[Union[Path, str]] = None) -> Dict[str, Any]:
    """Retrieve database metrics, file size, and record counts."""
    resolved_path = get_db_path(db_path)
    if not resolved_path.exists():
        return {
            "status": "uninitialized",
            "total_events": 0,
            "db_size_bytes": 0,
            "db_size_mb": 0.0,
            "oldest_event": None,
            "newest_event": None,
        }

    conn = get_db_connection(db_path)
    try:
        cursor = conn.execute(
            """
            SELECT COUNT(*) as total, MIN(timestamp) as oldest, MAX(timestamp) as newest
            FROM detection_events
            """
        )
        row = cursor.fetchone()
        size_bytes = resolved_path.stat().st_size
        return {
            "status": "ready",
            "db_path": str(resolved_path),
            "total_events": row["total"] if row else 0,
            "db_size_bytes": size_bytes,
            "db_size_mb": round(size_bytes / (1024 * 1024), 2),
            "oldest_event": row["oldest"] if row else None,
            "newest_event": row["newest"] if row else None,
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "total_events": 0,
            "db_size_bytes": 0,
            "db_size_mb": 0.0,
        }
    finally:
        conn.close()


def prune_events(
    max_age_days: float = 30.0,
    db_path: Optional[Union[Path, str]] = None,
) -> int:
    """Prune events older than max_age_days from SQLite."""
    cutoff_time = time.time() - (max_age_days * 86400.0)
    cutoff_str = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(cutoff_time))
    conn = get_db_connection(db_path)
    try:
        with conn:
            cursor = conn.execute("DELETE FROM detection_events WHERE timestamp < ?", (cutoff_str,))
            deleted = cursor.rowcount
        return deleted
    finally:
        conn.close()


def clear_events(db_path: Optional[Union[Path, str]] = None) -> int:
    """Clear all records from detection_events table (used for test isolation)."""
    conn = get_db_connection(db_path)
    try:
        with conn:
            cursor = conn.execute("DELETE FROM detection_events")
            deleted = cursor.rowcount
        return deleted
    finally:
        conn.close()


def save_session(session: Dict[str, Any], db_path: Optional[Union[Path, str]] = None) -> bool:
    """Insert or update a wildlife sighting session in SQLite."""
    conn = get_db_connection(db_path)
    try:
        with conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO sighting_sessions (
                    session_id, stream_id, species, start_time, end_time,
                    duration_seconds, detection_count, peak_confidence,
                    best_snapshot_url, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    session["session_id"],
                    session["stream_id"],
                    session["species"],
                    session["start_time"],
                    session["end_time"],
                    float(session.get("duration_seconds", 0.0)),
                    int(session.get("detection_count", 1)),
                    float(session.get("peak_confidence", 0.0)),
                    session.get("best_snapshot_url"),
                    session.get("status", "active"),
                ),
            )
        return True
    finally:
        conn.close()


def get_session_by_id(session_id: str, db_path: Optional[Union[Path, str]] = None) -> Optional[Dict[str, Any]]:
    """Fetch a single sighting session by its ID."""
    conn = get_db_connection(db_path)
    try:
        cursor = conn.execute(
            """
            SELECT session_id, stream_id, species, start_time, end_time,
                   duration_seconds, detection_count, peak_confidence,
                   best_snapshot_url, status
            FROM sighting_sessions
            WHERE session_id = ?
            """,
            (session_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None
        return dict(row)
    finally:
        conn.close()


def query_sessions(
    limit: int = 50,
    stream_id: Optional[str] = None,
    species: Optional[str] = None,
    status: Optional[str] = None,
    db_path: Optional[Union[Path, str]] = None,
) -> List[Dict[str, Any]]:
    """Query recent sighting sessions with optional stream, species, and status filters."""
    conn = get_db_connection(db_path)
    try:
        clauses = []
        params = []
        if stream_id:
            clauses.append("stream_id = ?")
            params.append(stream_id)
        if species:
            clauses.append("species = ?")
            params.append(species)
        if status:
            clauses.append("status = ?")
            params.append(status)

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"""
            SELECT session_id, stream_id, species, start_time, end_time,
                   duration_seconds, detection_count, peak_confidence,
                   best_snapshot_url, status
            FROM sighting_sessions
            {where_sql}
            ORDER BY start_time DESC
            LIMIT ?
        """
        params.append(limit)
        cursor = conn.execute(query, params)
        return [dict(r) for r in cursor.fetchall()]
    finally:
        conn.close()


def clear_sessions(db_path: Optional[Union[Path, str]] = None) -> int:
    """Clear all records from sighting_sessions table (used for test isolation)."""
    conn = get_db_connection(db_path)
    try:
        with conn:
            cursor = conn.execute("DELETE FROM sighting_sessions")
            deleted = cursor.rowcount
        return deleted
    finally:
        conn.close()


def save_review(review: Dict[str, Any], db_path: Optional[Union[Path, str]] = None) -> bool:
    """Persist or update a human reinforcement review decision."""
    conn = get_db_connection(db_path)
    try:
        review_id = review.get("review_id") or f"rev_{uuid.uuid4().hex[:8]}"
        event_id = review.get("event_id")
        stream_id = review.get("stream_id", "")
        proposed_species = review.get("proposed_species", "")
        confidence = float(review.get("confidence", 0.0))
        snapshot_url = review.get("snapshot_url")
        bbox = review.get("bbox") or review.get("bbox_json")
        bbox_json = json.dumps(bbox) if isinstance(bbox, (list, dict)) else (bbox or "[]")
        decision = review.get("decision", "accept").lower()
        confirmed_species = review.get("confirmed_species") or proposed_species
        notes = review.get("notes") or ""
        reviewer = review.get("reviewer", "Mike")
        timestamp = review.get("timestamp") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        with conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO human_reviews (
                    review_id, event_id, stream_id, proposed_species,
                    confidence, snapshot_url, bbox_json, decision,
                    confirmed_species, notes, reviewer, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    review_id,
                    event_id,
                    stream_id,
                    proposed_species,
                    confidence,
                    snapshot_url,
                    bbox_json,
                    decision,
                    confirmed_species,
                    notes,
                    reviewer,
                    timestamp,
                ),
            )
        return True
    except Exception as e:
        logger.error(f"Error saving human review {review.get('event_id')}: {e}")
        return False
    finally:
        conn.close()


def get_review_by_id(review_id: str, db_path: Optional[Union[Path, str]] = None) -> Optional[Dict[str, Any]]:
    """Retrieve a single review by review_id."""
    conn = get_db_connection(db_path)
    try:
        cursor = conn.execute("SELECT * FROM human_reviews WHERE review_id = ?", (review_id,))
        row = cursor.fetchone()
        if not row:
            return None
        res = dict(row)
        try:
            res["bbox"] = json.loads(res.get("bbox_json") or "[]")
        except Exception:
            res["bbox"] = []
        return res
    finally:
        conn.close()


def get_review_by_event_id(event_id: str, db_path: Optional[Union[Path, str]] = None) -> Optional[Dict[str, Any]]:
    """Retrieve a single review by event_id."""
    conn = get_db_connection(db_path)
    try:
        cursor = conn.execute("SELECT * FROM human_reviews WHERE event_id = ?", (event_id,))
        row = cursor.fetchone()
        if not row:
            return None
        res = dict(row)
        try:
            res["bbox"] = json.loads(res.get("bbox_json") or "[]")
        except Exception:
            res["bbox"] = []
        return res
    finally:
        conn.close()


def query_reviews(
    limit: int = 50,
    stream_id: Optional[str] = None,
    decision: Optional[str] = None,
    reviewer: Optional[str] = None,
    db_path: Optional[Union[Path, str]] = None,
) -> List[Dict[str, Any]]:
    """Query human review records with filtering."""
    conn = get_db_connection(db_path)
    try:
        clauses = []
        params = []
        if stream_id:
            clauses.append("stream_id = ?")
            params.append(stream_id)
        if decision:
            clauses.append("decision = ?")
            params.append(decision.lower())
        if reviewer:
            clauses.append("reviewer = ?")
            params.append(reviewer)

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"""
            SELECT * FROM human_reviews
            {where_sql}
            ORDER BY timestamp DESC
            LIMIT ?
        """
        params.append(limit)
        cursor = conn.execute(query, params)
        results = []
        for row in cursor.fetchall():
            d = dict(row)
            try:
                d["bbox"] = json.loads(d.get("bbox_json") or "[]")
            except Exception:
                d["bbox"] = []
            results.append(d)
        return results
    finally:
        conn.close()


def get_review_queue(
    limit: int = 50,
    stream_id: Optional[str] = None,
    min_confidence: float = 0.0,
    max_confidence: float = 1.0,
    db_path: Optional[Union[Path, str]] = None,
) -> List[Dict[str, Any]]:
    """Retrieve detection events that have not yet been reviewed by humans."""
    conn = get_db_connection(db_path)
    try:
        clauses = ["hr.review_id IS NULL"]
        params = []
        if stream_id:
            clauses.append("e.stream_id = ?")
            params.append(stream_id)
        if min_confidence > 0.0:
            clauses.append("e.confidence >= ?")
            params.append(min_confidence)
        if max_confidence < 1.0:
            clauses.append("e.confidence <= ?")
            params.append(max_confidence)

        where_sql = f"WHERE {' AND '.join(clauses)}"
        query = f"""
            SELECT e.event_id, e.timestamp, e.stream_id, e.species,
                   e.confidence, e.bbox_json, e.snapshot_url, e.metadata_json
            FROM detection_events e
            LEFT JOIN human_reviews hr ON e.event_id = hr.event_id
            {where_sql}
            ORDER BY e.timestamp DESC
            LIMIT ?
        """
        params.append(limit)
        cursor = conn.execute(query, params)
        results = []
        for r in cursor.fetchall():
            results.append(row_to_event_dict(r))
        return results
    finally:
        conn.close()


def get_review_stats(db_path: Optional[Union[Path, str]] = None) -> Dict[str, Any]:
    """Calculate aggregate stats for human reinforcement reviews."""
    conn = get_db_connection(db_path)
    try:
        # Total reviews
        cur = conn.execute("SELECT COUNT(*) as total FROM human_reviews")
        total = cur.fetchone()["total"]

        # Decisions
        cur = conn.execute(
            """
            SELECT decision, COUNT(*) as count
            FROM human_reviews
            GROUP BY decision
            """
        )
        decisions = {r["decision"]: r["count"] for r in cur.fetchall()}

        # Reviewers
        cur = conn.execute(
            """
            SELECT reviewer, COUNT(*) as count
            FROM human_reviews
            GROUP BY reviewer
            """
        )
        reviewers = {r["reviewer"]: r["count"] for r in cur.fetchall()}

        # By stream
        cur = conn.execute(
            """
            SELECT stream_id, COUNT(*) as count
            FROM human_reviews
            GROUP BY stream_id
            """
        )
        streams = {r["stream_id"]: r["count"] for r in cur.fetchall()}

        # Queue remaining count
        cur = conn.execute(
            """
            SELECT COUNT(*) as queue_count
            FROM detection_events e
            LEFT JOIN human_reviews hr ON e.event_id = hr.event_id
            WHERE hr.review_id IS NULL
            """
        )
        queue_count = cur.fetchone()["queue_count"]

        return {
            "total_reviews": total,
            "queue_remaining": queue_count,
            "accepted": decisions.get("accept", 0),
            "rejected": decisions.get("reject", 0),
            "relabelled": decisions.get("relabel", 0),
            "decisions": decisions,
            "reviewers": reviewers,
            "streams": streams,
        }
    finally:
        conn.close()


def clear_reviews(db_path: Optional[Union[Path, str]] = None) -> int:
    """Clear all records from human_reviews table (for test isolation)."""
    conn = get_db_connection(db_path)
    try:
        with conn:
            cursor = conn.execute("DELETE FROM human_reviews")
            return cursor.rowcount
    finally:
        conn.close()


