from graph import workflow
from graph.workflow import build_workflow


app = build_workflow()


result = app.invoke(
    {
        "question": "Can a Viewer create an API credential?",
        "retry_count": 0,
    }
)

print("\nClassification:")
print(result.get("classification"))

print("\nAnswer:")
print(result.get("answer"))

print("\nVerification:")
print(result.get("verification_passed"))

print("\nVerification reason:")
print(result.get("verification_reason"))

print("\nConfidence:")
print(result.get("confidence"))

print("\nRetry count:")
print(result.get("retry_count", 0))

print("\nRequires human:")
print(result.get("requires_human"))

print("\nReason:")
print(result.get("reason"))

print("\nSources:")

for source in result.get("sources", []):
    print("\nFile:", source["filename"])
    print("Excerpt:", source["excerpt"])
