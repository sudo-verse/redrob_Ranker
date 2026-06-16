#!/usr/bin/env python3
"""Redrob Ranker V1 — CLI entrypoint.

Reproduce command (Stage-3 compatible):

    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Runs CPU-only, no network, well under the 5-minute / 16 GB budget.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running as a plain script (`python rank.py`) from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ranker.config import load_config  # noqa: E402
from ranker.pipeline import run_pipeline  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Redrob candidate ranker (V1).")
    parser.add_argument(
        "--candidates", required=True, help="Path to candidates.jsonl (or .jsonl.gz)."
    )
    parser.add_argument(
        "--out", default="submission.csv", help="Output CSV path (default: submission.csv)."
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parent / "config.yaml"),
        help="Path to config.yaml (default: bundled config).",
    )
    parser.add_argument(
        "--log-level", default="INFO", help="Logging level (DEBUG/INFO/WARNING)."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = load_config(args.config)
    stats = run_pipeline(args.candidates, args.out, cfg)
    logging.getLogger("rank").info(
        "Done in %ss: %d candidates, %d honeypots rejected, top=%.4f cutoff=%.4f",
        stats["elapsed_seconds"],
        stats["total_candidates"],
        stats["honeypots_rejected"],
        stats["top_score"] or 0.0,
        stats["cutoff_score"] or 0.0,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
