#!/usr/bin/env python
"""Download the GGUF model from Hugging Face.

Usage::

    python scripts/download_model.py                # default: tencent/Youtu-LLM-2B (GGUF)
    python scripts/download_model.py --force        # re-download even if present
    python scripts/download_model.py --list-only    # just show found GGUF files
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lifelong_agent.llm.downloader import (  # noqa: E402
    DEFAULT_LOCAL_DIR,
    DEFAULT_REPO,
    DEFAULT_REVISION,
    ensure_model,
    find_gguf_files,
    select_best_gguf,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download the offline GGUF model")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--local-dir", default=DEFAULT_LOCAL_DIR)
    parser.add_argument("--force", action="store_true", help="Re-download if present")
    parser.add_argument("--list-only", action="store_true", help="Only list found files")
    args = parser.parse_args()

    if args.list_only:
        for path in find_gguf_files(Path(args.local_dir)):
            print(path)
        return 0

    try:
        gguf_path = ensure_model(
            repo=args.repo,
            revision=args.revision,
            local_dir=Path(args.local_dir),
            force=args.force,
        )
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Model ready: {gguf_path}")
    best = select_best_gguf(find_gguf_files(Path(args.local_dir)))
    print(f"Best GGUF file for inference: {best}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
