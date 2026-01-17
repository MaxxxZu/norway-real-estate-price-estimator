import os
import sys
import urllib.request


def main() -> int:
    port = os.getenv("REE_PORT", "8000")
    url = f"http://127.0.0.1:{port}/healthz"
    try:
        with urllib.request.urlopen(url, timeout=2) as resp:
            if 200 <= resp.status < 300:
                return 0
            print(f"Unexpected status: {resp.status}", file=sys.stderr)
            return 1
    except Exception as e:
        print(f"Healthcheck failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
