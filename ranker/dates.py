"""Small date helpers shared by the loader, honeypot detector, and features."""

from __future__ import annotations

from datetime import date


def parse_date(value: object) -> date | None:
    """Parse an ISO ``YYYY-MM-DD`` string; return ``None`` on anything invalid."""
    if not isinstance(value, str) or not value:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def months_between(start: date, end: date) -> int:
    """Whole months from ``start`` to ``end`` (negative if start is after end)."""
    return (end.year - start.year) * 12 + (end.month - start.month)
