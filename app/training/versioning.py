import os
import subprocess
from datetime import datetime, timezone


def _short_git_sha() -> str:
    sha = (os.getenv("GITHUB_SHA") or "").strip()
    if sha:
        return sha[:7]

    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        return out or "local"
    except Exception:
        return "local"


def make_model_version() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    return f"{ts}-{_short_git_sha()}"
