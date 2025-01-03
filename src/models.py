# src/models.py

from pydantic import BaseModel
from typing import List

class FactCheckResult(BaseModel):
    source: str
    relevance: float
    snippet: str

class FakeNewsAnalysis(BaseModel):
    image_text_match: bool
    fact_check_status: str  # "Supported" | "Contradicted" | "Inconclusive"
    is_fake_news: bool
    reasoning: str
    confidence_score: float
    fact_check_sources: List[FactCheckResult]
