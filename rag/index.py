from __future__ import annotations
import os, json
from typing import List, Dict, Tuple
import numpy as np
import faiss

def save_metadata(path: str, rows: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def load_metadata(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def build_faiss_index(vectors: np.ndarray) -> faiss.Index:
    # vectors are expected normalized; inner product ~= cosine similarity
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index

def persist_index(index_dir: str, index: faiss.Index, metadata: List[Dict]) -> None:
    os.makedirs(index_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(index_dir, "faiss.index"))
    save_metadata(os.path.join(index_dir, "metadata.jsonl"), metadata)

def load_index(index_dir: str) -> Tuple[faiss.Index, List[Dict]]:
    index = faiss.read_index(os.path.join(index_dir, "faiss.index"))
    metadata = load_metadata(os.path.join(index_dir, "metadata.jsonl"))
    return index, metadata
