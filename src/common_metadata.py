import subprocess
import time
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def get_runtime_metadata(script_path: str) -> dict:
    """Returns runtime provenance metadata for result artifacts."""
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_commit = "unknown"
    return {
        "generated_by": script_path,
        "git_commit": git_commit,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
    }

