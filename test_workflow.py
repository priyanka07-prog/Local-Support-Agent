from graph.workflow import build_workflow


app = build_workflow()


result = app.invoke(
    {
        "question": "Can a Viewer create an API credential?",
        "retry_count": 0,
    }
)


print("\nClassification:")
print(result["classification"])

print("\nRetrieved documents:")

for document in result["retrieved_documents"]:
    print("\nSource:", document["source"])
    print("Score:", round(document["score"], 4))
    print("Content:")
    print(document["content"][:300])
    print("-" * 60)