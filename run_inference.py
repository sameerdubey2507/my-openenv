"""
run_inference.py — Launcher for EMERGI-ENV inference.

Starts the FastAPI server on localhost:7860, waits for it to be healthy,
then runs inference.py with all arguments forwarded. Shuts down server after.

Usage:
    python run_inference.py                        # all 9 tasks, hybrid mode
    python run_inference.py --mode rule_based      # no LLM
    python run_inference.py --task task1_single_triage
    python run_inference.py --verbose --output results.json
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
import httpx
from pathlib import Path

ROOT = Path(__file__).parent
SERVER_PORT = int(os.getenv("PORT", "7860"))
SERVER_URL  = f"http://localhost:{SERVER_PORT}"
HEALTH_URL  = f"{SERVER_URL}/health"
MAX_WAIT_S  = 60                                             
CHECK_INTERVAL = 1.0

def wait_for_server(url: str, timeout: float = MAX_WAIT_S) -> bool:
    """Poll /health until the server responds or timeout."""
    deadline = time.monotonic() + timeout
    attempt = 0
    while time.monotonic() < deadline:
        attempt += 1
        try:
            r = httpx.get(url, timeout=3.0)
            if r.status_code < 500:
                print(f"  ✅  Server healthy after {attempt} attempts ({time.monotonic()-(deadline-timeout):.1f}s)")
                return True
        except Exception:
            pass
        print(f"  ⏳  Waiting for server... (attempt {attempt})", end="\r", flush=True)
        time.sleep(CHECK_INTERVAL)
    return False

def main() -> int:
                                                                               
    print()
    print("═" * 80)
    print("  EMERGI-ENV Launcher — starting FastAPI server on port", SERVER_PORT)
    print("═" * 80)

    server_proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "server.main:app",
            "--host", "0.0.0.0",
            "--port", str(SERVER_PORT),
            "--log-level", "warning",
        ],
        cwd=str(ROOT),
                                                            
        stdin=subprocess.DEVNULL,
    )

    print(f"\n  Waiting for server at {HEALTH_URL} …")
    if not wait_for_server(HEALTH_URL, timeout=MAX_WAIT_S):
        print(f"\n  ❌  Server did not become healthy within {MAX_WAIT_S}s. Aborting.")
        server_proc.terminate()
        return 1

    print()                                             

    inf_args = sys.argv[1:]
    inference_cmd = [sys.executable, str(ROOT / "inference.py")] + inf_args

    print("  Running:", " ".join(inference_cmd))
    print()

    try:
        result = subprocess.run(inference_cmd, cwd=str(ROOT))
        exit_code = result.returncode
    except KeyboardInterrupt:
        print("\n  ⚠️  Interrupted.")
        exit_code = 130
    finally:
                                                                               
        print("\n  Shutting down server …")
        server_proc.terminate()
        try:
            server_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server_proc.kill()

    return exit_code

if __name__ == "__main__":
    sys.exit(main())