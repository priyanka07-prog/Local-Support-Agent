from typing import Literal

from langgraph.graph import StateGraph, START, END

from graph.state import SupportState
from graph.nodes import triage_node, retrieve_node, generate_node


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
    """Retrieve relevant knowledge-base documents."""
    
    state = retrieve_node(state)
    state = generate_node(state)
    return state


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

    graph.add_edge("answerable", END)
    graph.add_edge("clarification", END)
    graph.add_edge("out_of_scope", END)

    return graph.compile()