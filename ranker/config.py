"""Typed configuration loading for the Redrob ranker (Stage 3 knobs)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Weights:
    title_fit: float = 25.0
    ir_depth: float = 20.0
    product_experience: float = 15.0
    availability: float = 15.0
    assessment_trust: float = 10.0
    location_fit: float = 5.0
    experience_fit: float = 5.0
    github_signal: float = 5.0

    def total(self) -> float:
        return (
            self.title_fit
            + self.ir_depth
            + self.product_experience
            + self.availability
            + self.assessment_trust
            + self.location_fit
            + self.experience_fit
            + self.github_signal
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "title_fit": self.title_fit,
            "ir_depth": self.ir_depth,
            "product_experience": self.product_experience,
            "availability": self.availability,
            "assessment_trust": self.assessment_trust,
            "location_fit": self.location_fit,
            "experience_fit": self.experience_fit,
            "github_signal": self.github_signal,
        }


@dataclass(frozen=True)
class HoneypotConfig:
    reference_date: str = "auto"
    tenure_slack_months: int = 2
    yoe_excess_buffer_months: int = 36


@dataclass(frozen=True)
class ScoringConfig:
    experience_band: tuple[float, float] = (5.0, 9.0)
    experience_decay_years: float = 6.0
    ir_depth_cap: float = 12.0
    recency_window_days: float = 180.0
    notice_period_cap_days: float = 120.0
    assessment_high_threshold: float = 60.0
    assessment_low_threshold: float = 25.0
    response_rate_floor: float = 0.40
    stuffer_penalty: float = 0.60
    stuffer_low_score_trigger: int = 3


@dataclass(frozen=True)
class OutputConfig:
    top_n: int = 100
    score_decimals: int = 6


@dataclass(frozen=True)
class Config:
    weights: Weights = field(default_factory=Weights)
    honeypot: HoneypotConfig = field(default_factory=HoneypotConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def _filtered(cls: type, data: dict[str, Any]) -> dict[str, Any]:
    """Keep only keys that ``cls`` accepts, so unknown YAML keys don't crash."""
    allowed = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    return {k: v for k, v in (data or {}).items() if k in allowed}


def load_config(path: str | Path | None) -> Config:
    """Load config from YAML, falling back to dataclass defaults for any gap."""
    if path is None:
        return Config()
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}

    weights = Weights(**_filtered(Weights, raw.get("weights", {})))
    honeypot = HoneypotConfig(**_filtered(HoneypotConfig, raw.get("honeypot", {})))

    scoring_raw = _filtered(ScoringConfig, raw.get("scoring", {}))
    if "experience_band" in scoring_raw:
        band = scoring_raw["experience_band"]
        scoring_raw["experience_band"] = (float(band[0]), float(band[1]))
    scoring = ScoringConfig(**scoring_raw)

    output = OutputConfig(**_filtered(OutputConfig, raw.get("output", {})))
    return Config(weights=weights, honeypot=honeypot, scoring=scoring, output=output)
