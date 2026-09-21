"""Outpost Snapshot Storage Retention & Pruning Engine.

Monitors snapshot storage footprint and enforces retention policies:
- Maximum age pruning (e.g. remove snapshots older than 24 hours)
- Maximum storage ceiling (e.g. prune oldest snapshots if storage > 500 MB)
"""

import argparse
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple


def get_snapshots_storage_stats(snapshots_dir: Path) -> Dict[str, float]:
    """Calculates total snapshot file count and total storage footprint in MB."""
    if not snapshots_dir.exists():
        return {"total_files": 0, "total_mb": 0.0}

    total_bytes = 0
    total_files = 0
    for root, _, files in os.walk(snapshots_dir):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                total_files += 1
                try:
                    total_bytes += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass

    return {
        "total_files": total_files,
        "total_mb": round(total_bytes / (1024 * 1024), 2),
    }


def prune_snapshots(
    snapshots_dir: Path,
    max_age_hours: float = 24.0,
    max_storage_mb: float = 500.0,
) -> Dict[str, Any]:
    """Prunes expired snapshots and enforces maximum storage ceiling."""
    if not snapshots_dir.exists():
        return {"pruned_files": 0, "freed_mb": 0.0}

    now = time.time()
    max_age_sec = max_age_hours * 3600.0

    # Collect all image files with mtime and size
    snapshot_files: List[Tuple[Path, float, int]] = []
    total_bytes = 0

    for root, _, files in os.walk(snapshots_dir):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                p = Path(root) / f
                try:
                    stat = p.stat()
                    snapshot_files.append((p, stat.st_mtime, stat.st_size))
                    total_bytes += stat.st_size
                except OSError:
                    pass

    pruned_files = 0
    freed_bytes = 0

    # Pass 1: Prune files older than max_age_hours
    surviving_files: List[Tuple[Path, float, int]] = []
    for p, mtime, size in snapshot_files:
        if (now - mtime) > max_age_sec:
            try:
                p.unlink()
                pruned_files += 1
                freed_bytes += size
                total_bytes -= size
            except OSError:
                surviving_files.append((p, mtime, size))
        else:
            surviving_files.append((p, mtime, size))

    # Pass 2: Enforce storage ceiling (FIFO - oldest first)
    max_bytes = int(max_storage_mb * 1024 * 1024)
    if total_bytes > max_bytes:
        surviving_files.sort(key=lambda x: x[1])  # Oldest first
        for p, mtime, size in surviving_files:
            if total_bytes <= max_bytes:
                break
            try:
                p.unlink()
                pruned_files += 1
                freed_bytes += size
                total_bytes -= size
            except OSError:
                pass

    # Clean up empty subdirectories
    for root, dirs, files in os.walk(snapshots_dir, topdown=False):
        for d in dirs:
            dir_path = Path(root) / d
            try:
                if not any(dir_path.iterdir()):
                    dir_path.rmdir()
            except OSError:
                pass

    return {
        "pruned_files": pruned_files,
        "freed_mb": round(freed_bytes / (1024 * 1024), 2),
        "remaining_mb": round(total_bytes / (1024 * 1024), 2),
    }


if __name__ == "__main__":
    default_snapshots_dir = str(Path(__file__).resolve().parent / "snapshots")
    parser = argparse.ArgumentParser(description="Outpost Snapshot Storage Pruning Engine")
    parser.add_argument("--dir", type=str, default=default_snapshots_dir)
    parser.add_argument("--max-age-hours", type=float, default=24.0)
    parser.add_argument("--max-storage-mb", type=float, default=500.0)
    args = parser.parse_args()

    target_dir = Path(args.dir)
    before_stats = get_snapshots_storage_stats(target_dir)
    print(f"[*] Snapshot Storage Before: {before_stats['total_files']} files ({before_stats['total_mb']} MB)")

    res = prune_snapshots(target_dir, args.max_age_hours, args.max_storage_mb)
    print(f"[*] Pruning Complete: Removed {res['pruned_files']} files ({res['freed_mb']} MB freed). Remaining: {res['remaining_mb']} MB")
