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
            "answer": "I could not find relevant information in the knowledge base.",
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
    retrieved knowledge-base documents.
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

    # Check 3: Retrieved documents have usable content
    valid_documents = [
        document
        for document in documents
        if document.get("content", "").strip()
    ]

    if not valid_documents:
        return {
            **state,
            "verification_passed": False,
            "verification_reason": (
                "Retrieved documents do not contain usable evidence."
            ),
            "confidence": 0.0,
        }

    # Combine the retrieved evidence.
    evidence = " ".join(
        document["content"].lower()
        for document in valid_documents
    )

    # Normalize answer words.
    answer_words = {
        word.strip(".,!?;:()[]{}\"'")
        for word in answer.lower().split()
        if len(word.strip(".,!?;:()[]{}\"'")) > 4
    }

    evidence_words = {
        word.strip(".,!?;:()[]{}\"'")
        for word in evidence.split()
        if len(word.strip(".,!?;:()[]{}\"'")) > 4
    }

    # Avoid division by zero.
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

    # Use the retrieval scores as another signal.
    retrieval_scores = [
        float(document.get("score", 0.0))
        for document in valid_documents
    ]

    best_retrieval_score = max(retrieval_scores)

    # Calculate a simple confidence score.
    confidence = (
        0.7 * grounding_ratio
        + 0.3 * max(0.0, min(1.0, best_retrieval_score))
    )

    confidence = round(
        max(0.0, min(1.0, confidence)),
        2,
    )

    # Minimum grounding requirement.
    passed = (
        grounding_ratio >= 0.25
        and best_retrieval_score >= 0.20
    )

    if passed:
        reason = (
            f"Answer is supported by retrieved evidence. "
            f"Grounding={grounding_ratio:.2f}, "
            f"retrieval_score={best_retrieval_score:.2f}."
        )
    else:
        reason = (
            f"Insufficient evidence support. "
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

    return {
        **state,
        "requires_human": state.get("requires_human", False),
        "reason": state.get(
            "reason",
            state.get("verification_reason", ""),
        ),
        "warnings": state.get("warnings", []),
    }