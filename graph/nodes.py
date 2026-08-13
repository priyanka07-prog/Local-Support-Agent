from unittest import result

from graph.state import SupportState
from graph.generator import LocalGenerator


def triage_node(state: SupportState) -> SupportState:
    """
    Classify the user's question into one of three categories:

    - answerable
    - needs_clarification
    - out_of_scope
    """

    question = state["question"].strip().lower()

    # Out-of-scope / unsafe requests
    out_of_scope_keywords = [
        "password",
        "secret key",
        "private key",
        "bypass",
        "hack",
        "exploit",
        "malware",
        "illegal",
    ]

    if any(keyword in question for keyword in out_of_scope_keywords):
        return {
            **state,
            "classification": "out_of_scope",
            "triage_reason": (
                "The request involves sensitive credentials, "
                "security bypasses, or unsupported activity."
            ),
            "requires_human": False,
            "reason": "Request is outside the supported assistance scope.",
        }

    # Questions that are too vague to answer reliably
    clarification_keywords = [
        "not working",
        "doesn't work",
        "does not work",
        "broken",
        "issue",
        "problem",
        "help",
    ]

    if any(keyword in question for keyword in clarification_keywords):
        return {
            **state,
            "classification": "needs_clarification",
            "triage_reason": (
                "The question does not contain enough specific "
                "information to identify the issue."
            ),
            "clarification_question": (
                "Could you describe what you are trying to do "
                "and what happens when it fails?"
            ),
            "requires_human": False,
            "reason": "More information is needed before troubleshooting.",
        }

    # Otherwise treat the question as potentially answerable.
    return {
        **state,
        "classification": "answerable",
        "triage_reason": "The question appears suitable for knowledge-base retrieval.",
        "requires_human": False,
        "reason": "The request can be handled using the knowledge base.",
    }
from utils.retriever import KnowledgeBaseRetriever


# Create the retriever once when the application starts.
retriever = KnowledgeBaseRetriever()


def retrieve_node(state: SupportState) -> SupportState:
    """Retrieve the most relevant knowledge-base documents."""

    question = state["question"]

    results = retriever.search(
        question,
        top_k=3,
    )

    retrieved_documents = []

    for result in results:
        retrieved_documents.append(
            {
                "source": result["source"],
                "content": result["content"],
                "score": result["score"],
            }
        )

    return {
        **state,
        "retrieved_documents": retrieved_documents,
    }
generator = LocalGenerator()

def generate_node(state: SupportState) -> SupportState:
    """Generate a grounded answer from the retrieved documents."""

    question = state["question"]
    documents = state.get("retrieved_documents", [])

    if not documents:
        return {
            **state,
            "answer": (
                "I could not find relevant information "
                "in the knowledge base."
            ),
            "sources": [],
        }

    answer = generator.generate(
        question=question,
        documents=documents,
    )

    sources = [
        {
            "filename": document["source"],
            "excerpt": document["content"][:300],
        }
        for document in documents
    ]

    return {
        **state,
        "answer": answer,
        "sources": sources,
    }
def verify_node(state: SupportState) -> SupportState:
    """
    Verify that the generated answer is supported by the
    retrieved knowledge-base evidence.

    This verifier checks:
    1. The answer exists.
    2. Evidence exists.
    3. The answer has meaningful overlap with the evidence.
    4. Obvious contradictions are detected.
    """

    answer = state.get("answer", "").strip()
    documents = state.get("retrieved_documents", [])

    # Check 1: Answer exists

    if not answer:
        return {
            **state,
            "verification_passed": False,
            "verification_reason": "The generated answer is empty.",
            "confidence": 0.0,
        }

    # Check 2: Evidence exists

    if not documents:
        return {
            **state,
            "verification_passed": False,
            "verification_reason": (
                "No knowledge-base evidence was retrieved."
            ),
            "confidence": 0.0,
        }

    # Combine evidence

    evidence = " ".join(
        document.get("content", "")
        for document in documents
    ).lower()

    answer_lower = answer.lower()

    # Check 3: Meaningful word overlap

    answer_words = {
        word.strip(".,!?;:()[]{}\"'")
        for word in answer_lower.split()
        if len(word.strip(".,!?;:()[]{}\"'")) > 4
    }

    evidence_words = {
        word.strip(".,!?;:()[]{}\"'")
        for word in evidence.split()
        if len(word.strip(".,!?;:()[]{}\"'")) > 4
    }

    if not answer_words:
        return {
            **state,
            "verification_passed": False,
            "verification_reason": (
                "The answer does not contain enough meaningful content."
            ),
            "confidence": 0.0,
        }

    overlap = answer_words.intersection(evidence_words)

    grounding_ratio = len(overlap) / len(answer_words)

    # Check 4: Detect obvious contradictions

    contradiction_detected = False
    contradiction_reason = ""

    positive_patterns = [
        "yes",
        "can create",
        "can do",
        "is allowed",
        "are allowed",
        "has permission",
        "have permission",
        "is permitted",
        "are permitted",
    ]

    negative_patterns = [
        "no",
        "cannot",
        "can't",
        "not allowed",
        "not permitted",
        "does not have permission",
        "do not have permission",
        "cannot create",
    ]

    answer_positive = any(
        pattern in answer_lower
        for pattern in positive_patterns
    )

    answer_negative = any(
        pattern in answer_lower
        for pattern in negative_patterns
    )

    evidence_positive = any(
        pattern in evidence
        for pattern in positive_patterns
    )

    evidence_negative = any(
        pattern in evidence
        for pattern in negative_patterns
    )

    # If the answer and evidence clearly disagree,
    # verification must fail.
    if answer_positive and evidence_negative:
        contradiction_detected = True
        contradiction_reason = (
            "The answer appears to contradict the retrieved evidence."
        )

    elif answer_negative and evidence_positive:
        contradiction_detected = True
        contradiction_reason = (
            "The answer appears to contradict the retrieved evidence."
        )

    # Retrieval score

    retrieval_scores = [
        float(document.get("score", 0.0))
        for document in documents
    ]

    best_retrieval_score = max(
        retrieval_scores,
        default=0.0,
    )

    # Confidence

    confidence = (
        0.7 * grounding_ratio
        + 0.3 * max(
            0.0,
            min(1.0, best_retrieval_score),
        )
    )

    # Contradictions should heavily reduce confidence.
    if contradiction_detected:
        confidence *= 0.2

    confidence = round(
        max(0.0, min(1.0, confidence)),
        2,
    )

    # Final verification decision

    passed = (
        grounding_ratio >= 0.25
        and best_retrieval_score >= 0.20
        and not contradiction_detected
    )

    if contradiction_detected:
        reason = contradiction_reason

    elif passed:
        reason = (
            "Answer is supported by the retrieved evidence. "
            f"Grounding={grounding_ratio:.2f}, "
            f"retrieval_score={best_retrieval_score:.2f}."
        )

    else:
        reason = (
            "The answer does not have sufficient support from "
            "the retrieved knowledge-base evidence. "
            f"Grounding={grounding_ratio:.2f}, "
            f"retrieval_score={best_retrieval_score:.2f}."
        )

    return {
        **state,
        "verification_passed": passed,
        "verification_reason": reason,
        "confidence": confidence,
    }

def final_response_node(state: SupportState) -> SupportState:
    """Prepare the final structured response."""

    classification = state.get("classification", "")
    answer = state.get("answer", "")

    sources = []

    for source in state.get("sources", []):
        sources.append(
            {
                "document": source.get(
                    "filename",
                    "unknown",
                ),
                "passage": source.get(
                    "excerpt",
                    "",
                ),
            }
        )

    return {
        **state,
        "classification": classification,
        "answer": answer,
        "sources": sources,
        "confidence": state.get("confidence", 0.0),
        "requires_human": state.get("requires_human", False),
         "reason": state.get(
            "reason",
            state.get(
                "verification_reason",
                "",
            ),
        ),
    }
    