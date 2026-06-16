"""Unit tests for Stage 1 — HoneypotDetector."""

from __future__ import annotations

import unittest
from datetime import date

from ranker.config import HoneypotConfig
from ranker.honeypot import HoneypotDetector
from tests.fixtures import make_candidate

REF = date(2026, 6, 1)


class HoneypotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.detector = HoneypotDetector(REF, HoneypotConfig())

    def test_clean_candidate_passes(self) -> None:
        verdict = self.detector.screen(make_candidate())
        self.assertFalse(verdict.is_honeypot)
        self.assertEqual(verdict.honeypot_score, 0)
        self.assertFalse(verdict.final_rank_excluded)

    def test_veto_a_impossible_tenure(self) -> None:
        cand = make_candidate(
            career_history=[
                {
                    "company": "Pied Piper",
                    "title": "Engineer",
                    "start_date": "2024-01-01",
                    "end_date": None,
                    "duration_months": 200,  # only ~29 months possible by REF
                    "is_current": True,
                    "industry": "Software",
                    "company_size": "51-200",
                    "description": "x",
                }
            ]
        )
        verdict = self.detector.screen(cand)
        self.assertTrue(verdict.is_honeypot)
        self.assertEqual(verdict.honeypot_score, 1)
        self.assertTrue(any("impossible_tenure" in r for r in verdict.reasons))

    def test_veto_b_expert_zero_months(self) -> None:
        cand = make_candidate(
            skills=[
                {"name": "Docker", "proficiency": "expert", "endorsements": 0, "duration_months": 0}
            ]
        )
        verdict = self.detector.screen(cand)
        self.assertTrue(verdict.is_honeypot)
        self.assertTrue(any("expert_zero_months" in r for r in verdict.reasons))

    def test_veto_c_yoe_inconsistency(self) -> None:
        cand = make_candidate(
            profile={"years_of_experience": 3.0},
            career_history=[
                {
                    "company": "Acme",
                    "title": "Engineer",
                    "start_date": "2010-01-01",
                    "end_date": "2024-01-01",
                    "duration_months": 168,  # 14y of roles vs 3y stated => excess > 36mo buffer
                    "is_current": False,
                    "industry": "Software",
                    "company_size": "201-500",
                    "description": "x",
                }
            ],
        )
        # Note: veto A may also fire here; we only assert C is present.
        verdict = self.detector.screen(cand)
        self.assertTrue(verdict.is_honeypot)
        self.assertTrue(any("yoe_excess" in r for r in verdict.reasons))

    def test_slack_allows_minor_rounding(self) -> None:
        cand = make_candidate(
            career_history=[
                {
                    "company": "Acme",
                    "title": "Engineer",
                    "start_date": "2024-01-01",
                    "end_date": None,
                    "duration_months": 30,  # ~29 elapsed + 2 slack => allowed
                    "is_current": True,
                    "industry": "Software",
                    "company_size": "201-500",
                    "description": "x",
                }
            ],
            profile={"years_of_experience": 7.0},
        )
        self.assertFalse(self.detector.screen(cand).is_honeypot)


if __name__ == "__main__":
    unittest.main()
