from graph.nodes import triage_node


questions = [
    "Can a Viewer create an API credential?",
    "Sync is not working.",
    "Can you give me someone's password?",
]


for question in questions:
    state = {
        "question": question,
        "retry_count": 0,
    }

    result = triage_node(state)

    print("\nQuestion:", question)
    print("Classification:", result["classification"])
    print("Reason:", result["reason"])

    if result.get("clarification_question"):
        print("Clarification:", result["clarification_question"])