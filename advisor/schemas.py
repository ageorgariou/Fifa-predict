"""Pydantic schemas for Claude advisor I/O. TODO(step 20)."""
from __future__ import annotations

from pydantic import BaseModel


class Advice(BaseModel):
    starting_xi: list[str]
    formation: str
    bench_order: list[str]
    captain: str
    vice_captain: str
    captain_reasoning: str
    key_risks: list[str]
    differentials: list[str]
    chip_recommendation: str | None
    overall_strategy: str
