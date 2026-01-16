from __future__ import annotations
from typing import List, Dict
from dotenv import load_dotenv
from openai import OpenAI

from .config import SETTINGS
from .embeddings import Embedder
from .index import load_index
from .retrieve import search
from .guardrails import filter_retrieved
from .generate import answer_with_citations

import time
from .logging import log_event

class RAGPipeline:
    def __init__(self, index_dir: str = "data/index"):
        load_dotenv()
        self.embedder = Embedder(SETTINGS.embed_model)
        self.index, self.metadata = load_index(index_dir)
        self.client = OpenAI()

    def ask(self, question: str) -> Dict:
        t0 = time.time()

        qvec = self.embedder.embed([question])[0]
        retrieved = search(self.index, self.metadata, qvec, SETTINGS.top_k)
        retrieved = filter_retrieved(retrieved)

        # similarity gate (cosine ~ inner product since normalized)
        if not retrieved or retrieved[0]["score"] < SETTINGS.min_score:
            return {
                "answer": "Not in the provided documents.",
                "sources": [],
                "retrieved": retrieved,
            }

        answer = answer_with_citations(self.client, SETTINGS.openai_model, question, retrieved)
        # simple post-check: if model forgot citations and didn't say "Not in..."
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

        lat = time.time() - t0
        log_event("logs/rag_logs.jsonl", question, answer, sources, lat)

        return {"answer": answer, "sources": sources, "retrieved": retrieved}
