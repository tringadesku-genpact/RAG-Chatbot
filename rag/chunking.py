from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Dict

@dataclass
class Chunk:
    text: str
    meta: Dict

def chunk_text(text: str, meta: Dict, chunk_chars: int, overlap: int) -> List[Chunk]:
    """Simple character-based chunker with overlap."""
    text = (text or "").strip()
    if not text:
        return []

    chunks: List[Chunk] = []
    start = 0
    cid = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_chars)
        chunk = text[start:end].strip()
        if chunk:
            m = dict(meta)
            m["chunk_id"] = cid
            chunks.append(Chunk(text=chunk, meta=m))
            cid += 1
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks
