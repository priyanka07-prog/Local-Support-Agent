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