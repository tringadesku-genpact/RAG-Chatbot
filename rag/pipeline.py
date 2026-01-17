from __future__ import annotations
from typing import Dict, Optional
import time

from dotenv import load_dotenv
from openai import OpenAI

from .config import SETTINGS
from .embeddings import Embedder
from .index import load_index
from .retrieve import search
from .guardrails import filter_retrieved
from .generate import answer_with_citations
from .logging import log_event


UNKNOWN_MSG = "Not in the provided documents."


class RAGPipeline:
    def __init__(self, index_dir: str = "data/index"):
        load_dotenv()
        self.embedder = Embedder(SETTINGS.embed_model)
        self.index, self.metadata = load_index(index_dir)
        self.client = OpenAI()

    def ask(self, question: str, doc_filter: Optional[str] = None) -> Dict:
        t0 = time.time()

        qvec = self.embedder.embed([question])[0]

        retrieved = search(
            self.index,
            self.metadata,
            qvec,
            SETTINGS.top_k,
            doc_contains=doc_filter,
        )
        retrieved = filter_retrieved(retrieved)

        # If retrieval is weak, skip LLM and return unknown
        if not retrieved or retrieved[0]["score"] < SETTINGS.min_score:
            answer = UNKNOWN_MSG
            log_event("logs/rag_logs.jsonl", question, answer, [], time.time() - t0)
            return {"answer": answer, "sources": [], "retrieved": []}

        sources = [
            {
                "doc_name": r.get("doc_name"),
                "page": r.get("page"),
                "chunk_id": r.get("chunk_id"),
                "score": r.get("score"),
            }
            for r in retrieved
        ]

        answer = answer_with_citations(self.client, SETTINGS.openai_model, question, retrieved)

        if ("[" not in answer) and (UNKNOWN_MSG not in answer):
            answer = UNKNOWN_MSG

        if answer.strip() == UNKNOWN_MSG:
            log_event("logs/rag_logs.jsonl", question, answer, [], time.time() - t0)
            return {"answer": answer, "sources": [], "retrieved": []}

        log_event("logs/rag_logs.jsonl", question, answer, sources, time.time() - t0)
        return {"answer": answer, "sources": sources, "retrieved": retrieved}
