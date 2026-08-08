from utils.retriever import KnowledgeBaseRetriever


retriever = KnowledgeBaseRetriever()

results = retriever.search(
    "Can a Viewer create an API credential?",
    top_k=3,
)

print("\nRetrieved documents:\n")

for result in results:
    print("Source:", result["source"])
    print("Score:", round(result["score"], 4))
    print("Content preview:", result["content"][:200])
    print("-" * 60)