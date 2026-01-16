from __future__ import annotations
import argparse, os
from typing import List, Dict
from dotenv import load_dotenv

from .config import SETTINGS
from .loaders import iter_documents
from .cleaning import clean_text
from .chunking import chunk_text
from .embeddings import Embedder
from .index import build_faiss_index, persist_index

def main():
    load_dotenv()

    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/raw")
    ap.add_argument("--index_dir", default="data/index")
    args = ap.parse_args()

    embedder = Embedder(SETTINGS.embed_model)

    all_texts: List[str] = []
    all_meta: List[Dict] = []

    for text, meta in iter_documents(args.data_dir):
        text = clean_text(text)
        # Chunk per page (for PDFs) or per file (md/txt)
        chunks = chunk_text(text, meta, SETTINGS.chunk_chars, SETTINGS.chunk_overlap)
        for ch in chunks:
            all_texts.append(ch.text)
            # store chunk text in metadata for retrieval + citations
            m = dict(ch.meta)
            m["text"] = ch.text
            all_meta.append(m)

    if not all_texts:
        raise SystemExit(f"No supported documents found in {args.data_dir} (pdf, md, txt).")

    vectors = embedder.embed(all_texts)
    index = build_faiss_index(vectors)
    persist_index(args.index_dir, index, all_meta)

    print(f"Indexed {len(all_texts)} chunks into {args.index_dir}")

if __name__ == "__main__":
    main()
