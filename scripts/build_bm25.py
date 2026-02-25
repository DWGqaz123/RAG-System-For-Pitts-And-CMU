"""
build_bm25.py
-------------
Build and query BM25 sparse indexes for each collection using bm25s.

One BM25 index per collection:
  - Reads  data/processed/{X}_chunks.jsonl
  - Writes data/processed/{X}_bm25/   (bm25s native format)

Index directory contents (managed by bm25s internally):
  data.csc.index.npy        — sparse score matrix (CSC)
  indices.csc.index.npy
  indptr.csc.index.npy
  vocab.index.json          — vocabulary
  params.index.json         — BM25 hyperparameters
  corpus.jsonl              — chunk metadata (chunk_id, section, etc.; no text)

The corpus stored inside the index is the chunk metadata dict (without the
heavy 'text' field).  At query time, results are returned as metadata dicts
augmented with a 'score' key; callers look up full text via chunk_id from
the original {X}_chunks.jsonl when needed.

Usage
-----
    from build_bm25 import build_bm25, bm25_search, load_bm25

    # Build (run once per collection)
    build_bm25("A")
    build_bm25("B")
    build_bm25("C")
    build_bm25("D")


    # Search (load index once, query many times)
    retriever = load_bm25("A")
    hits = bm25_search("A", "Carnegie Mellon robotics", top_k=5,
                        retriever=retriever)
    for h in hits:
        print(h["score"], h["chunk_id"], h["section"])
"""

import json
import re
import time
from pathlib import Path

import bm25s

# Paths & BM25 hyperparameters

PROCESSED_DIR = Path("data/processed")

# BM25 variant: "lucene" matches Elasticsearch / Lucene defaults (k1=1.2, b=0.75)
# and is well-tested on general text retrieval tasks.
BM25_METHOD = "lucene"
BM25_K1     = 1.2
BM25_B      = 0.75

# Metadata keys written into the corpus stored inside the index.
# 'text' is intentionally excluded to keep corpus.jsonl small.
CORPUS_KEYS = [
    "chunk_id", "doc_id", "collection",
    "source_url", "url", "md_title", "site_title",
    "section", "subsection", "chunk_index", "word_count",
]


# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------

def _preprocess(text: str) -> str:
    """
    Light preprocessing before BM25 tokenization:
      - Lowercase
      - Strip Markdown header prefixes (# ## ###)
      - Collapse whitespace
    bm25s.tokenize() handles stopword removal and splitting internally.
    """
    text = text.lower()
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def load_chunks(collection: str) -> list[dict]:
    """Load all chunks from data/processed/{collection}_chunks.jsonl."""
    path = PROCESSED_DIR / f"{collection}_chunks.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run the chunking pipeline first."
        )
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _index_dir(collection: str) -> Path:
    """Return the directory where the BM25 index for a collection is stored."""
    return PROCESSED_DIR / f"{collection}_bm25"


def build_bm25(collection: str) -> bm25s.BM25:
    """
    Build and save a BM25 index for the given collection.

    Steps
    -----
    1. Load chunks from {collection}_chunks.jsonl.
    2. Preprocess and tokenize the 'text' field of each chunk.
    3. Fit the BM25 index.
    4. Save index + metadata corpus to data/processed/{collection}_bm25/.

    Returns the fitted bm25s.BM25 retriever (ready for immediate search).
    """
    print("=" * 60)
    print(f"Building BM25 index — Collection {collection}")
    print("=" * 60)

    # 1. Load chunks
    chunks = load_chunks(collection)
    n = len(chunks)
    print(f"  Loaded {n} chunks")

    # 2. Preprocess texts and build metadata corpus
    texts     = [_preprocess(c["text"]) for c in chunks]
    corpus    = [{k: c.get(k) for k in CORPUS_KEYS} for c in chunks]

    # 3. Tokenize (bm25s built-in English stopword list)
    print("  Tokenizing...")
    t0        = time.time()
    tokenized = bm25s.tokenize(texts, stopwords="en", show_progress=False)
    print(f"  Tokenized in {time.time() - t0:.1f}s  "
          f"(vocab size: {len(tokenized.vocab):,})")

    # 4. Fit BM25 index
    print("  Indexing...")
    t0        = time.time()
    retriever = bm25s.BM25(
        method=BM25_METHOD,
        k1=BM25_K1,
        b=BM25_B,
        corpus=corpus,   # stored for save/load; not used during scoring
    )
    retriever.index(tokenized, show_progress=False)
    print(f"  Indexed in {time.time() - t0:.1f}s")

    # 5. Save to disk
    out_dir = _index_dir(collection)
    out_dir.mkdir(parents=True, exist_ok=True)
    retriever.save(str(out_dir), corpus=corpus)

    # Report sizes
    total_kb = sum(f.stat().st_size for f in out_dir.iterdir()) / 1024
    print(f"\n  → Saved to {out_dir}  ({total_kb:.0f} KB total)")
    for f in sorted(out_dir.iterdir()):
        print(f"      {f.name:<40} {f.stat().st_size / 1024:>7.1f} KB")

    return retriever


def load_bm25(collection: str) -> bm25s.BM25:
    """
    Load a previously built BM25 index from disk.

    Returns a bm25s.BM25 retriever ready for search.
    """
    out_dir = _index_dir(collection)
    if not out_dir.exists():
        raise FileNotFoundError(
            f"{out_dir} not found. Run build_bm25('{collection}') first."
        )
    retriever = bm25s.BM25.load(str(out_dir), load_corpus=True)
    vocab_size = len(retriever.vocab) if hasattr(retriever, "vocab") else "?"
    n_docs = len(retriever.corpus) if hasattr(retriever, "corpus") else "?"
    print(f"  Loaded BM25 index for Collection {collection}  "
          f"(vocab: {vocab_size}  docs: {n_docs})")
    return retriever


def bm25_search(
    query: str,
    collection: str | None = None,
    retriever: bm25s.BM25 | None = None,
    top_k: int = 5,
) -> list[dict]:
    """
    Search the BM25 index for a collection.

    Provide either:
      - collection  : loads the saved index from disk on every call (convenient
                      for one-off queries; use retriever= for repeated queries).
      - retriever   : a pre-loaded bm25s.BM25 object (faster for batched search).

    Returns
    -------
    List of metadata dicts sorted by BM25 score (descending), each augmented
    with a 'score' key.  The list has at most top_k entries; zero-score hits
    are excluded.
    """
    if retriever is None:
        if collection is None:
            raise ValueError("Provide either collection or retriever.")
        retriever = load_bm25(collection)

    # Preprocess and tokenize the query the same way as the corpus
    q_text    = _preprocess(query)
    q_tokens  = bm25s.tokenize([q_text], stopwords="en", show_progress=False)

    results, scores = retriever.retrieve(
        q_tokens,
        k=top_k,
        show_progress=False,
    )
    # results shape: (1, top_k) — one query
    hits: list[dict] = []
    for meta, score in zip(results[0], scores[0]):
        if float(score) <= 0:
            continue    # BM25 returns 0 for non-matching docs; skip them
        record = dict(meta)
        record["score"] = float(score)
        hits.append(record)

    return hits


# ---------------------------------------------------------------------------
# CLI convenience — run directly to build all collections
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Build BM25 sparse indexes for one or all collections"
    )
    parser.add_argument(
        "collections", nargs="*", default=["A", "B", "C", "D"],
        help="Collections to index (default: A B C D)"
    )
    args = parser.parse_args()
    for col in args.collections:
        build_bm25(col)
        print()


if __name__ == "__main__":
    main()