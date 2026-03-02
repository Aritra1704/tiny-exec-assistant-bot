import subprocess
import sys
import time
from pathlib import Path

WATCH_DIRS = (Path("src"), Path("tests"))
IGNORE_PARTS = {"__pycache__", ".git", ".venv", "venv"}
TEST_COMMAND = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-t", "."]


def iter_python_files():
    for watch_dir in WATCH_DIRS:
        if not watch_dir.exists():
            continue
        for path in watch_dir.rglob("*.py"):
            if any(part in IGNORE_PARTS for part in path.parts):
                continue
            yield path


def snapshot() -> dict[str, int]:
    return {str(path): path.stat().st_mtime_ns for path in iter_python_files()}


def run_tests() -> int:
    print("\nRunning test suite:", " ".join(TEST_COMMAND))
    completed = subprocess.run(TEST_COMMAND, check=False)
    status = "PASS" if completed.returncode == 0 else "FAIL"
    print(f"Test suite {status}")
    return completed.returncode


def main() -> int:
    print("Watching src/ and tests/ for Python changes. Press Ctrl-C to stop.")
    last_snapshot: dict[str, int] = {}

    try:
        while True:
            current_snapshot = snapshot()
            if current_snapshot != last_snapshot:
                run_tests()
                last_snapshot = current_snapshot
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopped test watcher.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
