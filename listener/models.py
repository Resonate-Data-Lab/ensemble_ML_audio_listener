"""Shared data structures for the Listener pipeline."""
from dataclasses import dataclass


@dataclass
class Recording:
    """An uploaded environmental sound recording (the source of truth)."""

    filename: str
    filepath: str
    duration_seconds: float
