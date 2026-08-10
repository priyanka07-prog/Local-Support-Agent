from graph.generator import LocalGenerator


generator = LocalGenerator()

answer = generator.generate(
    question="Can a Viewer create an API credential?",
    documents=[
        {
            "source": "05_api_credentials.md",
            "content": """
# API Credentials

OrbitDesk uses workspace API credentials for programmatic access.
Only Workspace Owners and Admins can create API credentials.
Viewers cannot create API credentials.
""",
        }
    ],
)

print("\nGenerated answer:")
print(answer)