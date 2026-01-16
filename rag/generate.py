from __future__ import annotations
from typing import List, Dict
from openai import OpenAI
from .prompt import SYSTEM_PROMPT, build_context

def answer_with_citations(client: OpenAI, model: str, question: str, retrieved: List[Dict]) -> str:
    context = build_context(retrieved)
    user_input = f"""Question: {question}

Context:
{context}
"""

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
        ],
    )
    return resp.output_text.strip()
