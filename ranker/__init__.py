"""Redrob candidate ranking system — V1 (deterministic, CPU-only).

Stages:
    0. loader      — streaming JSONL ingestion
    1. honeypot    — hard-veto rejection of impossible profiles
    2. features    — typed feature extraction
    3. scoring     — transparent weighted scoring engine
    4. explain     — grounded reasoning generation
    5. submission  — top-100 CSV emission
"""

from __future__ import annotations

__version__ = "1.0.0"
