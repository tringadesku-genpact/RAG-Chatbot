from __future__ import annotations
from typing import List, Dict, Tuple
import numpy as np
import faiss

def search(index: faiss.Index, metadata: List[Dict], query_vec: np.ndarray, top_k: int) -> List[Dict]:
    if query_vec.ndim == 1:
        query_vec = query_vec[None, :]
    scores, idxs = index.search(query_vec.astype("float32"), top_k)
    out: List[Dict] = []
    for score, idx in zip(scores[0].tolist(), idxs[0].tolist()):
        if idx == -1:
            continue
        row = dict(metadata[idx])
        row["score"] = float(score)
        out.append(row)
    return out
