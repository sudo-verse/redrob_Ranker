"""Unit tests for Stage 3/4 — scoring engine and explanations."""

from __future__ import annotations

import unittest
from datetime import date

from ranker.config import Config
from ranker.explain import build_reasoning
from ranker.features import extract
from ranker.scoring import score_candidate
from tests.fixtures import make_candidate

REF = date(2026, 6, 1)
CFG = Config()


def _score(cand: dict) -> float:
    f = extract(cand, CFG.scoring, REF)
    return score_candidate(f, CFG).score


class ScoringTests(unittest.TestCase):
    def test_score_in_unit_interval(self) -> None:
        for title in ["ML Engineer", "HR Manager", "Backend Engineer"]:
            s = _score(make_candidate(profile={"current_title": title}))
            self.assertGreaterEqual(s, 0.0)
            self.assertLessEqual(s, 1.0)

    def test_components_in_unit_interval(self) -> None:
        f = extract(make_candidate(), CFG.scoring, REF)
        result = score_candidate(f, CFG)
        for name, val in result.components.items():
            self.assertGreaterEqual(val, 0.0, name)
            self.assertLessEqual(val, 1.0, name)

    def test_strong_ai_beats_irrelevant(self) -> None:
        strong = _score(make_candidate())
        weak = _score(
            make_candidate(
                profile={"current_title": "HR Manager", "current_company": "Infosys",
                         "summary": "HR professional.", "location": "Indore, MP"},
                career_history=[
                    {"company": "Infosys", "title": "HR Manager", "start_date": "2018-01-01",
                     "end_date": None, "duration_months": 60, "is_current": True,
                     "industry": "IT Services", "company_size": "10001+", "description": "Recruiting."}
                ],
                skills=[{"name": "Excel", "proficiency": "advanced", "endorsements": 5, "duration_months": 40}],
                redrob_signals={"skill_assessment_scores": {}, "willing_to_relocate": False, "github_activity_score": -1},
            )
        )
        self.assertGreater(strong, weak)

    def test_consulting_penalizes(self) -> None:
        product = _score(make_candidate())
        consulting = _score(
            make_candidate(
                profile={"current_company": "Wipro"},
                career_history=[
                    {"company": "Wipro", "title": "ML Engineer", "start_date": "2020-01-01",
                     "end_date": None, "duration_months": 40, "is_current": True,
                     "industry": "IT Services", "company_size": "10001+",
                     "description": "Built a recommendation system."}
                ],
            )
        )
        self.assertGreater(product, consulting)

    def test_stuffer_penalty_applied(self) -> None:
        clean = extract(make_candidate(), CFG.scoring, REF)
        clean_result = score_candidate(clean, CFG)
        self.assertEqual(clean_result.stuffer_modifier, 1.0)

        stuffer_cand = make_candidate(
            profile={
                "summary": "lately curious how tools could augment work, experimented "
                "with chatgpt for productivity."
            }
        )
        f = extract(stuffer_cand, CFG.scoring, REF)
        result = score_candidate(f, CFG)
        self.assertLess(result.stuffer_modifier, 1.0)

    def test_reasoning_is_grounded(self) -> None:
        f = extract(make_candidate(), CFG.scoring, REF)
        result = score_candidate(f, CFG)
        text = build_reasoning(f, result)
        self.assertIn("ML Engineer", text)
        self.assertIn("Razorpay", text)  # real employer
        # Must not invent a skill the candidate lacks.
        self.assertNotIn("Kubernetes", text)
        self.assertTrue(text.endswith("."))

    def test_reasoning_flags_concerns(self) -> None:
        cand = make_candidate(
            redrob_signals={"recruiter_response_rate": 0.1, "notice_period_days": 90},
        )
        f = extract(cand, CFG.scoring, REF)
        result = score_candidate(f, CFG)
        text = build_reasoning(f, result)
        self.assertIn("Concerns", text)


if __name__ == "__main__":
    unittest.main()
