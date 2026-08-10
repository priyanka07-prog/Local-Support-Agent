from typing import Literal
from langgraph.graph import StateGraph, START, END
from graph.state import SupportState
from graph.nodes import (triage_node,
    retrieve_node,
    generate_node,
    verify_node)


def route_after_triage(
    state: SupportState,
) -> Literal["answerable", "needs_clarification", "out_of_scope"]:
    """Choose the next graph path based on the triage classification."""

    classification = state.get("classification")

    if classification == "answerable":
        return "answerable"

    if classification == "needs_clarification":
        return "needs_clarification"

    if classification == "out_of_scope":
        return "out_of_scope"

    raise ValueError(f"Unknown classification: {classification}")


def clarification_node(state: SupportState) -> SupportState:
    """Prepare the response for a clarification request."""

    return {
        **state,
        "answer": state.get(
            "clarification_question",
            "Could you provide more details about the issue?",
        ),
    }


def out_of_scope_node(state: SupportState) -> SupportState:
    """Prepare the response for an out-of-scope request."""

    return {
        **state,
        "answer": (
            "I can help with questions related to the supported "
            "OrbitDesk knowledge base, but I can't assist with this request."
        ),
    }


def answerable_node(state: SupportState) -> SupportState:
    """Regenerate the answer once failed verification."""
    
    retry_count = state.get("retry_count",0) + 1
    state = {
        **state,
        "retry_count":retry_count,
    }
    return generate_node(state)


def build_workflow():
    """Build and compile the LangGraph workflow."""

    graph = StateGraph(SupportState)

    graph.add_node("triage", triage_node)
    graph.add_node("answerable", answerable_node)
    graph.add_node("clarification", clarification_node)
    graph.add_node("out_of_scope", out_of_scope_node)

    graph.add_edge(START, "triage")

    graph.add_conditional_edges(
        "triage",
        route_after_triage,
        {
            "answerable": "answerable",
            "needs_clarification": "clarification",
            "out_of_scope": "out_of_scope",
        },
    )

    graph.add_edge("answerable", "verify")
    
    graph.add_conditional_edges(
        "verify",
        route_after_verification,
        {
            "pass" : END,
            "retry" : "retry_generate",
            "fail" : END
        },
    )
    graph.add_edge("retry_generate", "verify")
    graph.add_edge("clarification", END)
    graph.add_edge("out_of_scope", END)

    return graph.compile()
def route_after_verification(state: SupportState) -> str:
    """
    Decide whether the answer should be accepted or regenerated.
    """

    if state.get("verification_passed", False):
        return "pass"

    retry_count = state.get("retry_count", 0)

    if retry_count < 1:
        return "retry"

    return "fail"