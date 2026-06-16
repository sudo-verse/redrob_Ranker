"""Unit tests for the V2 retrieval-augmentation layer."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ranker.config import Config
from ranker.embeddings import HashingBackend, build_candidate_text
from ranker.pipeline_v2 import V2Config, _blend, run_pipeline_v2
from ranker.retrieval import build_jd_query_text, embed_query, retrieve_top_k
from ranker.embeddings import EmbeddingMatrix
from ranker.scoring import score_candidate
from ranker.features import extract
from ranker.semantic_evidence import extract_evidence, semantic_evidence_score
from tests.fixtures import make_candidate
from datetime import date

REF = date(2026, 6, 1)
CFG = Config()
V2CFG = V2Config(embedding_backend="hashing", retrieval_top_k=5)


class SemanticEvidenceTests(unittest.TestCase):
    def test_evidence_extracted_from_career(self) -> None:
        ev = extract_evidence(make_candidate(), semantic_similarity=0.5)
        self.assertIn("recommendation", ev.categories)
        self.assertTrue(ev.career_support)
        self.assertTrue(ev.title_support)
        self.assertGreater(semantic_evidence_score(ev), 0.0)

    def test_no_evidence_scores_zero(self) -> None:
        bare = make_candidate(
            profile={"summary": "Accountant managing ledgers.", "headline": "Accountant",
                     "current_title": "Accountant"},
            career_history=[{"company": "Acme", "title": "Accountant", "start_date": "2019-01-01",
                             "end_date": None, "duration_months": 40, "is_current": True,
                             "industry": "Finance", "company_size": "201-500",
                             "description": "Managed accounts payable and tax filings."}],
        )
        ev = extract_evidence(bare, 0.1)
        self.assertEqual(semantic_evidence_score(ev), 0.0)

    def test_score_bounded(self) -> None:
        ev = extract_evidence(make_candidate(), 0.9)
        self.assertGreaterEqual(semantic_evidence_score(ev), 0.0)
        self.assertLessEqual(semantic_evidence_score(ev), 1.0)


class BlendTests(unittest.TestCase):
    def test_v1_remains_dominant(self) -> None:
        f = extract(make_candidate(), CFG.scoring, REF)
        v1 = score_candidate(f, CFG)
        # max possible semantic lift vs zero lift.
        hi = _blend(v1.components, CFG, V2Config(semantic_weight=8.0), 1.0, v1.stuffer_modifier)
        lo = _blend(v1.components, CFG, V2Config(semantic_weight=8.0), 0.0, v1.stuffer_modifier)
        # The full-strength semantic component can move the score by at most
        # ~ w_sem/(100+w_sem) = 7.4%.
        self.assertLessEqual(hi - lo, 8.0 / 108.0 + 1e-9)
        self.assertGreaterEqual(hi - lo, 0.0)

    def test_zero_evidence_is_uniform_rescale_of_v1(self) -> None:
        # Adding a weighted component renormalizes by (100 + w_sem). With sem=0
        # every candidate is scaled by the SAME factor 100/108, so intra-group
        # ranking is preserved even though absolute scores shrink.
        f = extract(make_candidate(), CFG.scoring, REF)
        v1 = score_candidate(f, CFG)
        blended = _blend(v1.components, CFG, V2Config(semantic_weight=8.0), 0.0, v1.stuffer_modifier)
        expected = v1.score * (100.0 / 108.0)
        self.assertAlmostEqual(blended, expected, places=6)

    def test_order_preserved_for_unboosted_pair(self) -> None:
        strong = score_candidate(extract(make_candidate(), CFG.scoring, REF), CFG)
        weak = score_candidate(
            extract(make_candidate(profile={"current_title": "HR Manager"}), CFG.scoring, REF), CFG
        )
        b_strong = _blend(strong.components, CFG, V2Config(), 0.0, strong.stuffer_modifier)
        b_weak = _blend(weak.components, CFG, V2Config(), 0.0, weak.stuffer_modifier)
        self.assertEqual(strong.score > weak.score, b_strong > b_weak)


class RetrievalTests(unittest.TestCase):
    def test_query_and_retrieval(self) -> None:
        backend = HashingBackend()
        self.assertIn("recommendation", build_jd_query_text())
        cands = [make_candidate(candidate_id=f"CAND_{i:07d}") for i in range(1, 11)]
        texts = [build_candidate_text(c) for c in cands]
        matrix = backend.encode(texts)
        em = EmbeddingMatrix([c["candidate_id"] for c in cands], matrix, backend.name)
        retrieved = retrieve_top_k(em, backend, top_k=5)
        self.assertEqual(len(retrieved), 5)
        for sim in retrieved.values():
            self.assertGreaterEqual(sim, -1.0)
            self.assertLessEqual(sim, 1.0001)


class PipelineV2Tests(unittest.TestCase):
    def test_end_to_end_valid_submission(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            jsonl = Path(d) / "candidates.jsonl"
            out = Path(d) / "submission_v2.csv"
            with open(jsonl, "w", encoding="utf-8") as fh:
                for i in range(1, 121):
                    fh.write(json.dumps(make_candidate(candidate_id=f"CAND_{i:07d}")) + "\n")
            v2cfg = V2Config(embedding_backend="hashing", cache_dir=str(Path(d) / "cache"))
            records, stats = run_pipeline_v2(jsonl, out, CFG, v2cfg)
            self.assertEqual(stats["selected"], 100)
            self.assertEqual(len(records), 120)
            # every v2 score within [0,1]
            for r in records:
                self.assertGreaterEqual(r.v2_score, 0.0)
                self.assertLessEqual(r.v2_score, 1.0)


if __name__ == "__main__":
    unittest.main()
