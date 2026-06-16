"""Unit tests for Stage 2 — feature extraction."""

from __future__ import annotations

import unittest
from datetime import date

from ranker.config import ScoringConfig
from ranker.features import extract
from tests.fixtures import make_candidate

REF = date(2026, 6, 1)
SCORING = ScoringConfig()


class FeatureTests(unittest.TestCase):
    def test_direct_ai_title(self) -> None:
        f = extract(make_candidate(), SCORING, REF)
        self.assertTrue(f.direct_ai_title)
        self.assertFalse(f.adjacent_title)
        self.assertFalse(f.irrelevant_title)

    def test_irrelevant_title(self) -> None:
        f = extract(make_candidate(profile={"current_title": "HR Manager"}), SCORING, REF)
        self.assertTrue(f.irrelevant_title)
        self.assertFalse(f.direct_ai_title)

    def test_skill_counts_and_named_matches(self) -> None:
        f = extract(make_candidate(), SCORING, REF)
        self.assertIn("pytorch", f.matched_foundational)
        self.assertIn("learning to rank", f.matched_foundational)
        self.assertEqual(f.ranking_skill_count, len(f.matched_ranking))
        self.assertGreaterEqual(f.buzzword_skill_count, 2)  # pinecone + rag
        self.assertIn("pinecone", f.matched_vector_db)

    def test_company_signals(self) -> None:
        f = extract(make_candidate(), SCORING, REF)
        self.assertIn("razorpay", f.matched_product_companies)
        self.assertEqual(f.consulting_company_count, 0)

    def test_consulting_ratio(self) -> None:
        cand = make_candidate(
            career_history=[
                {"company": "Infosys", "title": "Engineer", "start_date": "2019-01-01",
                 "end_date": "2022-01-01", "duration_months": 36, "is_current": False,
                 "industry": "IT Services", "company_size": "10001+", "description": "x"},
                {"company": "TCS", "title": "Engineer", "start_date": "2022-02-01",
                 "end_date": None, "duration_months": 24, "is_current": True,
                 "industry": "IT Services", "company_size": "10001+", "description": "x"},
            ]
        )
        f = extract(cand, SCORING, REF)
        self.assertEqual(f.consulting_company_count, 2)
        self.assertAlmostEqual(f.consulting_ratio, 1.0)

    def test_text_evidence_detected(self) -> None:
        f = extract(make_candidate(), SCORING, REF)
        self.assertIn("recommendation system", f.text_evidence_phrases)
        self.assertIn("learning to rank", f.text_evidence_phrases)

    def test_assessment_trust_backed(self) -> None:
        f = extract(make_candidate(), SCORING, REF)
        # PyTorch (85) and Learning to Rank (78) both >= 60 and claimed advanced.
        self.assertEqual(f.assessment_backed_skills, 2)
        self.assertEqual(f.advanced_skill_low_score_count, 0)

    def test_assessment_low_score_flagged(self) -> None:
        cand = make_candidate(
            skills=[{"name": "NLP", "proficiency": "advanced", "endorsements": 1, "duration_months": 12}],
            redrob_signals={"skill_assessment_scores": {"NLP": 10}},
        )
        f = extract(cand, SCORING, REF)
        self.assertEqual(f.advanced_skill_low_score_count, 1)

    def test_recency_and_location(self) -> None:
        f = extract(make_candidate(), SCORING, REF)
        self.assertEqual(f.last_active_recency_days, (REF - date(2026, 5, 20)).days)
        self.assertTrue(f.preferred_location)
        self.assertTrue(f.relocatable)

    def test_dabbler_summary_detected(self) -> None:
        cand = make_candidate(
            profile={
                "summary": "5 professional years experience. lately curious how tools "
                "could augment my work, experimented with chatgpt for productivity."
            }
        )
        f = extract(cand, SCORING, REF)
        self.assertTrue(f.dabbler_summary)

    def test_github_sentinel(self) -> None:
        f = extract(make_candidate(redrob_signals={"github_activity_score": -1}), SCORING, REF)
        self.assertFalse(f.github_present)


if __name__ == "__main__":
    unittest.main()
