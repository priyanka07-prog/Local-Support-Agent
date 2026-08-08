from pathlib import Path
from typing import List, Dict, Any

from sentence_transformers import SentenceTransformer
import faiss


class KnowledgeBaseRetriever:
    def __init__(
        self,
        knowledge_base_path: str = "data/knowledge_base",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.knowledge_base_path = Path(knowledge_base_path)
        self.model = SentenceTransformer(model_name)

        self.documents: List[Dict[str, Any]] = []
        self.index = None

        self._load_documents()
        self._build_index()

    def _load_documents(self) -> None:
        """Load all Markdown knowledge-base files."""

        for file_path in sorted(self.knowledge_base_path.glob("*.md")):
            text = file_path.read_text(encoding="utf-8").strip()

            if text:
                self.documents.append(
                    {
                        "source": file_path.name,
                        "content": text,
                    }
                )

        if not self.documents:
            raise ValueError(
                f"No Markdown files found in {self.knowledge_base_path}"
            )

    def _build_index(self) -> None:
        """Create a FAISS index from document embeddings."""

        texts = [doc["content"] for doc in self.documents]

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings.astype("float32"))

    def search(
        self,
        query: str,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """Return the most relevant knowledge-base documents."""

        if not query.strip():
            return []

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        scores, indices = self.index.search(
            query_embedding.astype("float32"),
            min(top_k, len(self.documents)),
        )

        results = []

        for score, index in zip(scores[0], indices[0]):
            if index == -1:
                continue

            document = self.documents[index].copy()
            document["score"] = float(score)
            results.append(document)

        return results