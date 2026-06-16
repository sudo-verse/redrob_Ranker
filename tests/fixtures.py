"""Synthetic candidate builders for unit tests."""

from __future__ import annotations

from typing import Any


def make_candidate(**overrides: Any) -> dict:
    """A valid, schema-shaped candidate; override any field via kwargs."""
    base = {
        "candidate_id": "CAND_0000001",
        "profile": {
            "anonymized_name": "Test User",
            "headline": "ML Engineer",
            "summary": "Built recommendation systems and semantic search at a product company.",
            "location": "Pune, Maharashtra",
            "country": "India",
            "years_of_experience": 7.0,
            "current_title": "ML Engineer",
            "current_company": "Razorpay",
            "current_company_size": "201-500",
            "current_industry": "Fintech",
        },
        "career_history": [
            {
                "company": "Razorpay",
                "title": "ML Engineer",
                "start_date": "2021-01-01",
                "end_date": None,
                "duration_months": 30,
                "is_current": True,
                "industry": "Fintech",
                "company_size": "201-500",
                "description": "Built a recommendation system and learning to rank pipeline.",
            }
        ],
        "education": [
            {
                "institution": "IIT",
                "degree": "B.Tech",
                "field_of_study": "Computer Science",
                "start_year": 2014,
                "end_year": 2018,
                "grade": "8.5",
                "tier": "tier_1",
            }
        ],
        "skills": [
            {"name": "PyTorch", "proficiency": "advanced", "endorsements": 10, "duration_months": 36},
            {"name": "Learning to Rank", "proficiency": "advanced", "endorsements": 8, "duration_months": 30},
            {"name": "BM25", "proficiency": "intermediate", "endorsements": 5, "duration_months": 24},
            {"name": "Pinecone", "proficiency": "intermediate", "endorsements": 3, "duration_months": 12},
            {"name": "RAG", "proficiency": "intermediate", "endorsements": 4, "duration_months": 10},
        ],
        "certifications": [],
        "languages": [{"language": "English", "proficiency": "professional"}],
        "redrob_signals": {
            "profile_completeness_score": 90,
            "signup_date": "2021-01-01",
            "last_active_date": "2026-05-20",
            "open_to_work_flag": True,
            "profile_views_received_30d": 40,
            "applications_submitted_30d": 5,
            "recruiter_response_rate": 0.8,
            "avg_response_time_hours": 4.0,
            "skill_assessment_scores": {"PyTorch": 85, "Learning to Rank": 78},
            "connection_count": 300,
            "endorsements_received": 50,
            "notice_period_days": 15,
            "expected_salary_range_inr_lpa": {"min": 25, "max": 40},
            "preferred_work_mode": "hybrid",
            "willing_to_relocate": True,
            "github_activity_score": 70,
            "search_appearance_30d": 20,
            "saved_by_recruiters_30d": 5,
            "interview_completion_rate": 0.9,
            "offer_acceptance_rate": 0.5,
            "verified_email": True,
            "verified_phone": True,
            "linkedin_connected": True,
        },
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = {**base[key], **value}
        else:
            base[key] = value
    return base
