from __future__ import annotations
import os
from typing import Dict, List, Iterator, Tuple
import fitz  # PyMuPDF

def load_pdf(path: str) -> Iterator[Tuple[str, Dict]]:
    """Yield (page_text, meta) for each page."""
    doc = fitz.open(path)
    name = os.path.basename(path)
    for i in range(len(doc)):
        page = doc[i]
        text = page.get_text("text")
        meta = {"doc_name": name, "doc_path": path, "page": i + 1, "source_type": "pdf"}
        yield text, meta

def load_text_file(path: str) -> Iterator[Tuple[str, Dict]]:
    name = os.path.basename(path)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    meta = {"doc_name": name, "doc_path": path, "page": None, "source_type": "text"}
    yield text, meta

def iter_documents(data_dir: str) -> Iterator[Tuple[str, Dict]]:
    for root, _, files in os.walk(data_dir):
        for fn in files:
            p = os.path.join(root, fn)
            ext = os.path.splitext(fn.lower())[1]
            if ext == ".pdf":
                yield from load_pdf(p)
            elif ext in (".md", ".txt"):
                yield from load_text_file(p)
            else:
                continue
