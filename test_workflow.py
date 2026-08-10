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

print("\nAnswer:")
print(result["answer"])

print("\nSources:")

for source in result.get("sources", []):
    print("\nFile:", source["filename"])
    print("Excerpt:", source["excerpt"])