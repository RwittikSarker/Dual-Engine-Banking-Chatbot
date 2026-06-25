"""
escalation.py
-------------
Rule-based escalation engine.

Recommends human escalation when:
  - Classifier confidence is below threshold
  - Retrieval similarity is below threshold
  - Query is likely out-of-domain
  - Generated answer looks too short / uninformative
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.utils import get_logger

logger = get_logger(__name__)

# ──────────────────────────────────────────────
# Out-of-domain signals
# ──────────────────────────────────────────────

_OOD_PATTERNS = [
    r"\bweather\b",
    r"\bsport(s)?\b",
    r"\bpolitics\b",
    r"\brecipe\b",
    r"\bcooking\b",
    r"\bmusic\b",
    r"\bmovie(s)?\b",
    r"\bfilm(s)?\b",
    r"\bnews\b",
    r"\bhealth\b",
    r"\bmedical\b",
    r"\bjoke(s)?\b",
    r"\bfunny\b",
]

_OOD_REGEX = re.compile("|".join(_OOD_PATTERNS), flags=re.IGNORECASE)


def _is_out_of_domain(query: str) -> bool:
    """Heuristic check for out-of-domain queries."""
    return bool(_OOD_REGEX.search(query))


def _is_weak_generation(answer: str) -> bool:
    """Flag very short or repetitive answers as weak."""
    stripped = answer.strip()
    if len(stripped.split()) < 5:
        return True
    # Detect generic fallback phrases
    fallback_phrases = [
        "i don't know",
        "i cannot answer",
        "i'm not sure",
        "no information",
    ]
    lower = stripped.lower()
    return any(phrase in lower for phrase in fallback_phrases)


# ──────────────────────────────────────────────
# Escalation result
# ──────────────────────────────────────────────

@dataclass
class EscalationResult:
    recommend: bool
    reasons: list[str]

    @property
    def recommendation(self) -> str:
        return "Yes" if self.recommend else "No"


# ──────────────────────────────────────────────
# Main evaluator
# ──────────────────────────────────────────────

class EscalationEngine:
    """Determines whether a query should be escalated to a human agent."""

    def __init__(
        self,
        confidence_threshold: float = 0.60,
        similarity_threshold: float = 0.50,
    ):
        self.confidence_threshold = confidence_threshold
        self.similarity_threshold = similarity_threshold

    def evaluate(
        self,
        query: str,
        confidence_score: float,
        retrieval_similarity: float,
        generated_answer: str,
    ) -> EscalationResult:
        """
        Evaluate all signals and return an EscalationResult.

        Parameters
        ----------
        query                : raw user query
        confidence_score     : classifier softmax probability of top class
        retrieval_similarity : cosine similarity of best retrieved example
        generated_answer     : text produced by the generator
        """
        reasons: list[str] = []

        if confidence_score < self.confidence_threshold:
            reasons.append(
                f"Low classifier confidence ({confidence_score:.2f} < {self.confidence_threshold})"
            )

        if retrieval_similarity < self.similarity_threshold:
            reasons.append(
                f"Low retrieval similarity ({retrieval_similarity:.2f} < {self.similarity_threshold})"
            )

        if _is_out_of_domain(query):
            reasons.append("Query appears out-of-domain")

        if _is_weak_generation(generated_answer):
            reasons.append("Generated answer is uninformative or too short")

        recommend = len(reasons) > 0

        if recommend:
            logger.info("Escalation recommended. Reasons: %s", reasons)
        else:
            logger.debug("No escalation needed.")

        return EscalationResult(recommend=recommend, reasons=reasons)


def build_escalation_engine(cfg: dict) -> EscalationEngine:
    """Instantiate EscalationEngine from config."""
    esc_cfg = cfg["escalation"]
    return EscalationEngine(
        confidence_threshold=esc_cfg["confidence_threshold"],
        similarity_threshold=esc_cfg["similarity_threshold"],
    )
