from typing import Literal

from langgraph.graph import StateGraph, START, END

from graph.state import SupportState
from graph.nodes import (
    triage_node,
    retrieve_node,
    generate_node,
    verify_node,
    final_response_node,
)


def route_after_triage(
    state: SupportState,
) -> Literal[
    "answerable",
    "needs_clarification",
    "out_of_scope",
]:
    classification = state.get("classification")

    if classification == "answerable":
        return "answerable"

    if classification == "needs_clarification":
        return "needs_clarification"

    if classification == "out_of_scope":
        return "out_of_scope"

    raise ValueError(
        f"Unknown classification: {classification}"
    )


def clarification_node(state: SupportState) -> SupportState:
    return {
        **state,
        "answer": state.get(
            "clarification_question",
            "Could you provide more details about the issue?",
        ),
    }


def out_of_scope_node(state: SupportState) -> SupportState:
    return {
        **state,
        "answer": (
            "I can help with questions related to the "
            "supported OrbitDesk knowledge base, but I "
            "can't assist with this request."
        ),
    }


def answerable_node(state: SupportState) -> SupportState:
    """Retrieve evidence and generate an answer."""

    state = retrieve_node(state)
    state = generate_node(state)

    return state


def retry_generation_node(state: SupportState) -> SupportState:
    """Regenerate the answer once after failed verification."""

    retry_count = state.get("retry_count", 0) + 1

    state = {
        **state,
        "retry_count": retry_count,
    }

    return generate_node(state)


def route_after_verification(state: SupportState) -> str:
    """Decide whether to pass, retry, or escalate."""

    if state.get("verification_passed", False):
        return "pass"

    retry_count = state.get("retry_count", 0)

    if retry_count < 1:
        return "retry"

    return "fail"


def escalation_node(state: SupportState) -> SupportState:
    """Handle answers that could not be sufficiently verified."""

    return {
        **state,
        "requires_human": True,
        "reason": (
            "The generated answer could not be sufficiently "
            "verified against the retrieved knowledge-base evidence."
        ),
        "answer": (
            "I'm unable to provide a sufficiently verified answer "
            "from the available knowledge base. This request should "
            "be reviewed by a human support agent."
        ),
    }


def build_workflow():
    """Build and compile the LangGraph workflow."""

    graph = StateGraph(SupportState)

    # -----------------------------
    # Register nodes
    # -----------------------------

    graph.add_node("triage", triage_node)

    graph.add_node(
        "answerable",
        answerable_node,
    )

    graph.add_node(
        "clarification",
        clarification_node,
    )

    graph.add_node(
        "out_of_scope",
        out_of_scope_node,
    )

    graph.add_node(
        "verify",
        verify_node,
    )

    graph.add_node(
        "retry_generation",
        retry_generation_node,
    )

    graph.add_node(
        "escalation",
        escalation_node,
    )

    graph.add_node(
        "final_response",
        final_response_node,
    )

    # START → TRIAGE

    graph.add_edge(
        START,
        "triage",
    )

    # TRIAGE → routing

    graph.add_conditional_edges(
        "triage",
        route_after_triage,
        {
            "answerable": "answerable",
            "needs_clarification": "clarification",
            "out_of_scope": "out_of_scope",
        },
    )

    # ANSWERABLE → VERIFY

    graph.add_edge(
        "answerable",
        "verify",
    )

    # VERIFY → PASS / RETRY / FAIL

    graph.add_conditional_edges(
        "verify",
        route_after_verification,
        {
            "pass": "final_response",
            "retry": "retry_generation",
            "fail": "escalation",
        },
    )

    # RETRY → VERIFY

    graph.add_edge(
        "retry_generation",
        "verify",
    )

    # ESCALATION → END

    graph.add_edge(
        "escalation",
        "final_response",
    )
    # OTHER PATHS → END
    graph.add_edge(
        "clarification",
        "final_response",
    )

    graph.add_edge(
        "out_of_scope",
        "final_response",
    )

    graph.add_edge(
        "final_response",
        END,
    )

    return graph.compile()