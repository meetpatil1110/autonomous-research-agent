"""Pydantic request/response models for the API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class ResearchRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
