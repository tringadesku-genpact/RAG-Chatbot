from __future__ import annotations
import re
from typing import List, Dict

INJECTION_PATTERNS = [
    r"ignore (all|any|previous) instructions",
    r"system prompt",
    r"developer message",
    r"jailbreak",
    r"do anything now",
]

_inj = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)

def looks_like_injection(text: str) -> bool:
    return bool(_inj.search(text or ""))

def filter_retrieved(chunks: List[Dict]) -> List[Dict]:
    # We don't drop content by default; we just don't obey instructions in it (prompt already says that).
    # Still, you can drop chunks that *are mostly* prompt-injection attempts.
    out = []
    for c in chunks:
        if looks_like_injection(c.get("text","")):
            continue
        out.append(c)
    return out
