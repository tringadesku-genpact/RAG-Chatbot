import os
from dataclasses import dataclass

@dataclass(frozen=True)
class Settings:
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-5.2")
    embed_model: str = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    chunk_chars: int = int(os.getenv("CHUNK_CHARS", "1100"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "180"))
    top_k: int = int(os.getenv("TOP_K", "6"))
    min_score: float = float(os.getenv("MIN_SCORE", "0.25"))  # cosine similarity threshold (approx)

SETTINGS = Settings()
