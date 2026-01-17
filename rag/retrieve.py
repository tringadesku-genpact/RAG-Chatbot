from __future__ import annotations
from typing import List, Dict, Optional
import numpy as np
import faiss


def search(
    index: faiss.Index,
    metadata: List[Dict],
    query_vec: np.ndarray,
    top_k: int,
    doc_contains: Optional[str] = None,
    source_type: Optional[str] = None,
) -> List[Dict]:
    """
    FAISS vector search with optional metadata filtering.

    - doc_contains: keep only results whose doc_name contains this substring (case-insensitive)
    - source_type: keep only results whose source_type equals this value (e.g., 'pdf', 'text')
    """

    if query_vec.ndim == 1:
        query_vec = query_vec[None, :]

    # Pull more than top_k because filters can remove matches
    scores, idxs = index.search(query_vec.astype("float32"), top_k * 3)

    out: List[Dict] = []
    for score, idx in zip(scores[0].tolist(), idxs[0].tolist()):
        if idx == -1:
            continue

        row = dict(metadata[idx])

        # ✅ Metadata filters
        if doc_contains and doc_contains.lower() not in row.get("doc_name", "").lower():
            continue
        if source_type and row.get("source_type") != source_type:
            continue

        row["score"] = float(score)
        out.append(row)

        if len(out) >= top_k:
            break

    return out
