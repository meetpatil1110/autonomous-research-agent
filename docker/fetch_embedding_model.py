"""Pre-fetches chromadb's default embedding model at Docker build time.

Uses a plain urlretrieve rather than chromadb's own httpx-based downloader,
whose default timeout is too short for a slow or jittery connection to a
~80MB file. This only needs to land the archive at chromadb's expected
cache path — chromadb itself does the SHA256 check and extraction on
first use, so a partial/corrupt download here just gets re-fetched then.
"""
from __future__ import annotations

import time
import urllib.request
from pathlib import Path

URL = "https://chroma-onnx-models.s3.amazonaws.com/all-MiniLM-L6-v2/onnx.tar.gz"
DEST = Path.home() / ".cache/chroma/onnx_models/all-MiniLM-L6-v2/onnx.tar.gz"


def main() -> None:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 6):
        try:
            urllib.request.urlretrieve(URL, DEST)
            return
        except OSError as exc:
            print(f"download attempt {attempt} failed: {exc}")
            time.sleep(3)
    raise SystemExit("failed to download embedding model after 5 attempts")


if __name__ == "__main__":
    main()
