#!/usr/bin/env python3
"""Redrob Ranker V2 — CLI entrypoint (retrieval-augmented).

V2 adds a semantic-evidence component (~7%) on top of the untouched V1 scoring
engine. Embeddings are a one-time pre-computation cached to disk; the ranking
step stays well under the 5-minute budget.

    python rank_v2.py --candidates ./candidates.jsonl --out ./submission_v2.csv [--analyze]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ranker.analysis_v2 import compare  # noqa: E402
from ranker.config import load_config  # noqa: E402
from ranker.pipeline_v2 import load_v2_config, run_pipeline_v2  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Redrob candidate ranker (V2, retrieval-augmented).")
    p.add_argument("--candidates", required=True, help="Path to candidates.jsonl(.gz).")
    p.add_argument("--out", default="submission_v2.csv", help="Output CSV path.")
    p.add_argument("--config", default=str(Path(__file__).resolve().parent / "config.yaml"))
    p.add_argument("--analyze", action="store_true", help="Emit a V1-vs-V2 comparison JSON.")
    p.add_argument("--analysis-out", default="v2_comparison.json")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = load_config(args.config)
    v2cfg = load_v2_config(args.config)
    records, stats = run_pipeline_v2(args.candidates, args.out, cfg, v2cfg)
    logging.getLogger("rank_v2").info("V2 done: %s", stats)

    if args.analyze:
        cmp = compare(records, cfg.output.top_n)
        payload = {
            "stats": stats,
            "n_added": len(cmp.added),
            "n_removed": len(cmp.removed),
            "n_rescues": len(cmp.rescues),
            "n_potential_false_positives": len(cmp.potential_false_positives),
            "added": [
                {"id": r.candidate_id, "title": r.title, "v1": round(r.v1_score, 6),
                 "v2": round(r.v2_score, 6), "sem": round(r.semantic_evidence, 4),
                 "snippets": r.snippets}
                for r in cmp.added
            ],
            "removed": [
                {"id": r.candidate_id, "title": r.title, "v1": round(r.v1_score, 6),
                 "v2": round(r.v2_score, 6)}
                for r in cmp.removed
            ],
            "rescues": [
                {"id": r.candidate_id, "title": r.title, "sem": round(r.semantic_evidence, 4),
                 "sim": round(r.semantic_similarity, 4), "snippets": r.snippets}
                for r in cmp.rescues
            ],
        }
        Path(args.analysis_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logging.getLogger("rank_v2").info("Wrote analysis to %s", args.analysis_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
