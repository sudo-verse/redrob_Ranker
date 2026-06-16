"""Unit tests for Stage 5 — submission generation and end-to-end pipeline."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from ranker.config import Config
from ranker.pipeline import run_pipeline
from ranker.submission import ScoredCandidate, rank_and_select, write_submission
from tests.fixtures import make_candidate


class SubmissionTests(unittest.TestCase):
    def test_rank_and_select_ordering(self) -> None:
        cands = [
            ScoredCandidate("CAND_0000003", 0.5, "x"),
            ScoredCandidate("CAND_0000001", 0.9, "x"),
            ScoredCandidate("CAND_0000002", 0.5, "x"),  # tie with 0003 -> id asc
        ]
        ordered = rank_and_select(cands, top_n=3)
        self.assertEqual(
            [c.candidate_id for c in ordered],
            ["CAND_0000001", "CAND_0000002", "CAND_0000003"],
        )

    def test_write_requires_enough_rows(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                write_submission([ScoredCandidate("CAND_0000001", 0.5, "x")],
                                 Path(d) / "out.csv", top_n=100, score_decimals=6)

    def test_end_to_end_pipeline(self) -> None:
        # 120 distinct synthetic candidates + 1 honeypot.
        with tempfile.TemporaryDirectory() as d:
            jsonl = Path(d) / "candidates.jsonl"
            out = Path(d) / "submission.csv"
            with open(jsonl, "w", encoding="utf-8") as fh:
                for i in range(1, 121):
                    cand = make_candidate(candidate_id=f"CAND_{i:07d}")
                    fh.write(json.dumps(cand) + "\n")
                # one honeypot (expert + 0 months) -> must be excluded.
                hp = make_candidate(
                    candidate_id="CAND_9999999",
                    skills=[{"name": "Go", "proficiency": "expert", "endorsements": 0,
                             "duration_months": 0}],
                )
                fh.write(json.dumps(hp) + "\n")

            stats = run_pipeline(jsonl, out, Config())
            self.assertEqual(stats["honeypots_rejected"], 1)
            self.assertEqual(stats["selected"], 100)

            with open(out, encoding="utf-8") as fh:
                rows = list(csv.reader(fh))
            self.assertEqual(rows[0], ["candidate_id", "rank", "score", "reasoning"])
            data = rows[1:]
            self.assertEqual(len(data), 100)
            # honeypot never appears
            ids = {r[0] for r in data}
            self.assertNotIn("CAND_9999999", ids)
            # ranks 1..100 unique
            ranks = [int(r[1]) for r in data]
            self.assertEqual(sorted(ranks), list(range(1, 101)))
            # scores monotonically non-increasing
            scores = [float(r[2]) for r in data]
            self.assertTrue(all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1)))
            # equal-score rows ordered by candidate_id ascending
            for i in range(len(data) - 1):
                if scores[i] == scores[i + 1]:
                    self.assertLess(data[i][0], data[i + 1][0])


if __name__ == "__main__":
    unittest.main()
