from typing import TypedDict, List, Dict, Any, Optional


class SupportState(TypedDict, total=False):
    question: str

    classification: str
    triage_reason: str

    retrieved_documents: List[Dict[str, Any]]

    answer: str

    sources: List[Dict[str, str]]

    verification_passed: bool
    verification_reason: str

    retry_count: int

    confidence: float
    requires_human: bool
    reason: str
    clarification_question: Optional[str]
    warnings: List[str]