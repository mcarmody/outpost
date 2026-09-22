"""Restarts the three live Outpost edge runners (cornell/anacapa/katmai) on
whatever --model checkpoint is given. Windows-only -- this is the edge box's
own management script, not something the Railway-hosted server ever runs.

Exists so train_finetune.py --auto-deploy has something to actually call
once a new checkpoint clears its deploy gate; also fine to run by hand:

  python restart_runners.py --model models/active.pt
  python restart_runners.py --model yolo11x.pt   # revert to the base model

Uses the same two things proven reliable tonight (2026-09-22) launching
these runners over SSH: killing/creating processes via WMI/CIM rather than
Start-Process, because a process Start-Process spawns from an SSH exec
session gets reaped when that SSH session's job object closes -- a WMI
Win32_Process.Create call doesn't inherit that job and survives.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Mirrors KNOWN_STREAMS in server.py. Duplicated rather than imported
# because this script runs standalone on the edge box, which doesn't need
# (and for cv2/streamlink/ultralytics reasons, doesn't have) the server's
# own dependency set installed.
STREAMS = [
    {"stream_id": "cornell_feeder_01", "url": "https://www.youtube.com/watch?v=x10vL6_47Dw"},
    {"stream_id": "anacapa_kelp_01", "url": "https://www.youtube.com/watch?v=OAJF1Ie1m_Q"},
    {"stream_id": "katmai_brooks_01", "url": "https://www.youtube.com/watch?v=J7ZrIDvqlic"},
]


def kill_existing_runners() -> None:
    ps = (
        "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" "
        "| Where-Object { $_.CommandLine -like '*runner.py*' } "
        "| ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=False)


def launch_runner(stream_id: str, url: str, model: str, api_url: str, min_confidence: float) -> None:
    cwd = str(BASE_DIR)
    log_name = f"{stream_id}.restart.log"
    cmd_line = (
        f'cmd /c cd /d {cwd} && python -u runner.py '
        f'--url {url} --stream-id {stream_id} --model "{model}" '
        f'--api-url {api_url} --min-confidence {min_confidence} '
        f'> {log_name} 2>&1'
    )
    ps = (
        "$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create "
        f"-Arguments @{{CommandLine='{cmd_line}'}}; "
        "Write-Output \"ReturnValue=$($r.ReturnValue) PID=$($r.ProcessId)\""
    )
    result = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True)
    print(f"[*] {stream_id}: {result.stdout.strip()}")
    if result.returncode != 0:
        print(f"[!] {stream_id}: {result.stderr.strip()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, required=True, help="Path (or bare name, e.g. yolo11x.pt) to run all three runners on")
    parser.add_argument("--api-url", type=str, default="https://outpost.brock.ventures")
    parser.add_argument("--min-confidence", type=float, default=0.40)
    args = parser.parse_args()

    print("[*] Killing existing runner.py processes...")
    kill_existing_runners()
    time.sleep(3)

    print(f"[*] Launching 3 runners on model={args.model} ...")
    for s in STREAMS:
        launch_runner(s["stream_id"], s["url"], args.model, args.api_url, args.min_confidence)

    time.sleep(3)
    check = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -like '*runner.py*' }).Count"],
        capture_output=True, text=True,
    )
    count = (check.stdout or "0").strip()
    print(f"[*] {count} runner.py process(es) confirmed alive.")
    if count != "3":
        print("[!] Expected 3 live runners, got a different count -- check the individual *.restart.log files.")
        sys.exit(1)


if __name__ == "__main__":
    main()
