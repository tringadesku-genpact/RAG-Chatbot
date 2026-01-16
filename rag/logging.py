from __future__ import annotations
import os, json, time
from datetime import datetime
from typing import Dict, List

def log_event(log_path: str, question: str, answer: str, sources: List[Dict], latency_s: float):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    row = {
        "ts": datetime.utcnow().isoformat(),
        "question": question,
        "answer_preview": (answer or "")[:200],
        "num_sources": len(sources),
        "top_source": (sources[0]["doc_name"] if sources else None),
        "latency_s": round(latency_s, 3),
        "unknown": (answer.strip() == "Not in the provided documents."),
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
