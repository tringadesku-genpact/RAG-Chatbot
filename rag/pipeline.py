from __future__ import annotations
from typing import Dict, Optional
from dotenv import load_dotenv
from openai import OpenAI

from .config import SETTINGS
from .embeddings import Embedder
from .index import load_index
from .retrieve import search
from .guardrails import filter_retrieved
from .generate import answer_with_citations


class RAGPipeline:
    def __init__(self, index_dir: str = "data/index"):
        load_dotenv()
        self.embedder = Embedder(SETTINGS.embed_model)
        self.index, self.metadata = load_index(index_dir)
        self.client = OpenAI()

    def ask(self, question: str, doc_filter: Optional[str] = None) -> Dict:
        qvec = self.embedder.embed([question])[0]

        retrieved = search(
            self.index,
            self.metadata,
            qvec,
            SETTINGS.top_k,
            doc_contains=doc_filter,
        )

        retrieved = filter_retrieved(retrieved)

        if not retrieved or retrieved[0]["score"] < SETTINGS.min_score:
            return {
                "answer": "Not in the provided documents.",
                "sources": [],
                "retrieved": retrieved,
            }

        answer = answer_with_citations(self.client, SETTINGS.openai_model, question, retrieved)

        if ("[" not in answer) and ("Not in the provided documents" not in answer):
            answer = "Not in the provided documents."

        sources = [
            {
                "doc_name": r.get("doc_name"),
                "page": r.get("page"),
                "chunk_id": r.get("chunk_id"),
                "score": r.get("score"),
            }
            for r in retrieved
        ]

        return {"answer": answer, "sources": sources, "retrieved": retrieved}
