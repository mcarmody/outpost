"""Threshold gate for the Outpost auto-retrain loop.

Mike, #side-project 2026-09-22 18:50 PT: "Pick a data account threshold at
which you think it would be appropriate to deploy an update [to] the
model with the new training and then implement that threshold."

Chosen thresholds, and why:
- INITIAL_THRESHOLD = 100 total accept/relabel review examples before the
  first auto-deploy is even attempted. YOLO fine-tuning on a base model
  can produce *something* readable from a couple dozen examples (see the
  27-example proof-of-run tonight, mAP50 0.23 -- functional pipeline, not
  a usable model), but object-detection fine-tunes generally want on the
  order of dozens of examples per class before the numbers mean anything.
  100 total, split across however many species have accumulated by then,
  is a reasonable floor for "this might actually be better than the base
  model" rather than "this technically ran."
- RETRAIN_INCREMENT = 50: once a model IS deployed, don't retrain again
  until another 50 examples have come in. Retraining on every single new
  /review decision would mean an almost-continuous GPU load for a marginal
  data delta each time, and constant runner restarts (each one drops
  in-flight detections for a few seconds). 50 is enough new data to
  plausibly move the needle again without retraining every hour.
- The per-class (--min-per-class 15) and quality (--min-map50 0.50) gates
  live in train_finetune.py itself, not here -- this script only decides
  WHEN to bother attempting a run; train_finetune.py decides whether the
  result of that attempt is good enough to actually go live. A run that
  clears the count threshold here but fails the quality gate there is
  expected and fine: the checkpoint is kept for inspection, the runners
  keep serving whatever was already live, and this script's own state
  still advances (so it doesn't retry on the identical dataset every
  cycle -- it'll pick back up once another RETRAIN_INCREMENT of examples
  land).

Meant to be invoked on a schedule (see the Windows Scheduled Task created
alongside this -- OutpostAutoRetrain, daily). Also fine to run by hand.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parent
STATE_PATH = BASE_DIR / "models" / "retrain_state.json"

INITIAL_THRESHOLD = 100
RETRAIN_INCREMENT = 50


def load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except Exception:
            pass
    return {"last_attempted_count": 0, "last_deployed_count": 0, "last_run_at": None}


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2))


def current_review_count(api_url: str) -> int:
    resp = requests.get(f"{api_url.rstrip('/')}/api/review/export", params={"format": "json"}, timeout=15.0)
    resp.raise_for_status()
    return int(resp.json().get("count", 0))


def should_attempt(current: int, state: dict, force: bool = False) -> "tuple[bool, str]":
    """Pure decision logic, split out from main() so it's unit-testable
    without mocking the network or a subprocess call."""
    if force:
        return True, "forced"
    if current < INITIAL_THRESHOLD:
        return False, f"{current} < INITIAL_THRESHOLD={INITIAL_THRESHOLD}"
    last_attempted = state.get("last_attempted_count", 0)
    since_last = current - last_attempted
    if last_attempted > 0 and since_last < RETRAIN_INCREMENT:
        return False, f"only {since_last} new example(s) since last attempt (need {RETRAIN_INCREMENT})"
    return True, "threshold cleared"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", type=str, default="https://outpost.brock.ventures")
    parser.add_argument("--force", action="store_true", help="Skip the threshold checks and run train_finetune.py --auto-deploy unconditionally")
    args = parser.parse_args()

    state = load_state()
    current = current_review_count(args.api_url)
    print(f"[*] Current reviewed (accept+relabel) example count: {current}")
    print(f"[*] State: {state}")

    attempt, reason = should_attempt(current, state, force=args.force)
    if not attempt:
        print(f"[*] {reason}. Nothing to do.")
        return

    print(f"[*] {reason} -- running train_finetune.py --auto-deploy")
    train_script = BASE_DIR / "train_finetune.py"
    result = subprocess.run([sys.executable, str(train_script), "--api-url", args.api_url, "--auto-deploy"])

    state["last_attempted_count"] = current
    state["last_run_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if result.returncode == 0:
        state["last_deployed_count"] = current
        print("[*] Deployed.")
    elif result.returncode == 3:
        print("[*] Trained but did not clear the deploy gate -- not deployed. Will try again once more data lands.")
    else:
        print(f"[!] train_finetune.py exited {result.returncode} (unexpected -- check its output above).")
    save_state(state)


if __name__ == "__main__":
    main()
