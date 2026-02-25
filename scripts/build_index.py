"""
build_index.py
--------------
Generate embeddings for all chunks in a collection using
BAAI/bge-m3 and build a FAISS IndexFlatIP (cosine) index.

IndexFlatIP on L2-normalised vectors = cosine similarity search.

Model: BAAI/bge-m3
  - 560 M parameters
  - Embedding dim  : 1024
  - Max seq length : 8192 tokens
  - Public model   : no HuggingFace token required
  - Dense retrieval via sentence-transformers (used here)
  - No special passage prompt prefix needed (only queries need one;
    see search() below)

Usage (import in notebook):
    from build_index import build_index
    build_index("A")

Output files (written to data/processed/):
    A_index.faiss        — FAISS IndexFlatIP
    A_index_meta.jsonl   — chunk metadata (same order as FAISS vectors)
"""

import json
import platform
import time
from pathlib import Path

import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer


# Config

PROCESSED_DIR = Path("data/processed")

MODEL_NAME = "BAAI/bge-m3"
NORMALIZE  = True    # L2-normalise → IndexFlatIP becomes cosine similarity

BATCH_SIZE = 8

# Prefix added for bge-m3 asymmetric retrieval (queries only)
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# 2000 chars ≈ 400-500 tokens — well within model limits and MPS-safe.
MAX_SEQ_CHARS = 2000


# Device detection

def _detect_device() -> str:
    """
    Return the best available torch device string.
    """
    if platform.system() == "Darwin" and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# Helpers

def load_chunks(collection: str) -> list[dict]:
    """Load chunks from data/processed/<collection>_chunks.jsonl."""
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


def _load_model(device: str) -> SentenceTransformer:
    """
    Load BAAI/bge-m3 onto the given device.
    """
    print(f"  Loading model : {MODEL_NAME}")
    print(f"  Device        : {device.upper()}")
    t0 = time.time()

    model = SentenceTransformer(MODEL_NAME, device=device)

    print(f"  Model loaded in {time.time() - t0:.1f}s")
    return model


def _truncate_texts(texts: list[str], max_chars: int) -> list[str]:
    """Truncate each text to max_chars characters to avoid MPS OOM on long chunks."""
    truncated = [t[:max_chars] for t in texts]
    long_count = sum(1 for t in texts if len(t) > max_chars)
    if long_count:
        print(f"  [truncate] {long_count}/{len(texts)} chunks truncated "
              f"to {max_chars} chars")
    return truncated


def _encode_on_device(
    texts: list[str],
    model: SentenceTransformer,
    device: str,
    batch_size: int,
    normalize: bool,
) -> np.ndarray:
    """
    Encode texts on the given device, halving batch_size on OOM.
    If batch_size reaches 1 and still OOMs, raises RuntimeError
    so the caller can fall back to CPU.
    """
    current_batch = batch_size
    while current_batch >= 1:
        try:
            embeddings = model.encode(
                texts,
                batch_size=current_batch,
                show_progress_bar=True,
                normalize_embeddings=normalize,
                convert_to_numpy=True,
            )
            return embeddings.astype("float32")

        except RuntimeError as exc:
            oom_kws = ("invalid buffer size", "out of memory",
                       "mps backend", "cuda out of memory")
            if not any(k in str(exc).lower() for k in oom_kws):
                raise  # unrelated — re-raise immediately

            new_batch = max(1, current_batch // 2)
            print(f"  [OOM/{device.upper()}] batch_size={current_batch}. "
                  f"Retrying with batch_size={new_batch}...")

            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()

            if new_batch == current_batch:
                # Already at batch_size=1 and still OOMing
                raise RuntimeError(
                    f"OOM at batch_size=1 on {device.upper()}. "
                    "Will fall back to CPU."
                ) from exc
            current_batch = new_batch

    raise RuntimeError("encode_chunks: batch_size reached 0")


def encode_chunks(
    chunks: list[dict],
    model: SentenceTransformer,
    batch_size: int = BATCH_SIZE,
    normalize: bool = NORMALIZE,
) -> np.ndarray:
    """
    Encode the 'text' field of each chunk (passage side).

    Returns a float32 ndarray of shape (n_chunks, 1024).
    Vectors are L2-normalised when normalize=True so that IndexFlatIP
    gives cosine similarity scores in [-1, 1].

    Robustness strategy:
      1. Truncate each chunk to MAX_SEQ_CHARS chars before encoding
         (prevents single-chunk OOM even at batch_size=1).
      2. Try MPS/CUDA with the given batch_size, halving on OOM.
      3. If MPS/CUDA still OOMs at batch_size=1, reload model on CPU
         and encode there (slow but always succeeds).

    bge-m3 asymmetric retrieval note:
      Passages are encoded WITHOUT a prefix. Only queries need QUERY_PREFIX.
    """
    texts = _truncate_texts([c["text"] for c in chunks], MAX_SEQ_CHARS)
    n = len(texts)
    device = model.device.type if hasattr(model.device, "type") else str(model.device)

    print(f"  Encoding {n} chunks on {device.upper()} (batch_size={batch_size})...")
    t0 = time.time()

    try:
        embeddings = _encode_on_device(texts, model, device, batch_size, normalize)

    except RuntimeError:
        # Final fallback: CPU encoding — always succeeds, just slower
        print("  [fallback] Switching to CPU for encoding...")
        cpu_model = SentenceTransformer(MODEL_NAME, device="cpu")
        embeddings = cpu_model.encode(
            texts,
            batch_size=max(1, batch_size * 4),  # CPU can handle larger batches
            show_progress_bar=True,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
        ).astype("float32")

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s  ({n / elapsed:.1f} chunks/s)")
    return embeddings


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build a FAISS IndexFlatIP from pre-normalised float32 embeddings.
    IndexFlatIP on L2-normalised vectors = exact cosine similarity.
    faiss-cpu runs fully on CPU — no GPU needed on macOS.
    """
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def save_index(
    index: faiss.IndexFlatIP,
    chunks: list[dict],
    collection: str,
) -> tuple[Path, Path]:
    """
    Persist the FAISS index and a parallel metadata JSONL to data/processed/.

    The i-th metadata line corresponds to the i-th FAISS vector.
    The heavy 'text' field is omitted to keep the metadata file small;
    retrieve it from <collection>_chunks.jsonl by chunk_id when needed.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    index_path = PROCESSED_DIR / f"{collection}_index.faiss"
    meta_path  = PROCESSED_DIR / f"{collection}_index_meta.jsonl"

    faiss.write_index(index, str(index_path))

    meta_keys = [
        "chunk_id", "doc_id", "collection", "source_url", "url",
        "md_title", "site_title", "section", "subsection",
        "chunk_index", "word_count",
    ]
    with open(meta_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            record = {k: chunk.get(k) for k in meta_keys}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\n  → Index : {index_path}  ({index_path.stat().st_size / 1024:.0f} KB)")
    print(f"  → Meta  : {meta_path}  ({meta_path.stat().st_size  / 1024:.0f} KB)")
    return index_path, meta_path


# ──────────────────────────────────────────────
# Main entry points
# ──────────────────────────────────────────────

def build_index(
    collection: str,
    model_name: str = MODEL_NAME,
    batch_size: int = BATCH_SIZE,
) -> tuple[faiss.IndexFlatIP, list[dict]]:
    """
    Full pipeline: load chunks → encode passages → build FAISS index → save.

    Args:
        collection:  Single uppercase letter, e.g. "A".
        model_name:  HuggingFace model id (default: BAAI/bge-m3).
        batch_size:  Encoding batch size.

    Returns:
        (index, chunks) — in-memory FAISS index and the original chunk list.
    """
    print("=" * 60)
    print(f"Building FAISS index — Collection {collection}")
    print(f"Model : {model_name}")
    print("=" * 60)

    device = _detect_device()

    # 1. Load chunks
    chunks = load_chunks(collection)
    print(f"  Loaded {len(chunks)} chunks")

    # 2. Load model — no HF token needed for bge-m3
    model = _load_model(device)

    # 3. Encode passages (no prefix)
    embeddings = encode_chunks(chunks, model, batch_size=batch_size)
    print(f"  Embedding shape : {embeddings.shape}  dtype: {embeddings.dtype}")

    # 4. Sanity-check norms (should all be ~1.0 after normalisation)
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"  Vector norms    : mean={norms.mean():.4f}  "
          f"min={norms.min():.4f}  max={norms.max():.4f}")

    # 5. Build FAISS index
    index = build_faiss_index(embeddings)
    print(f"  FAISS index     : {index.__class__.__name__}  "
          f"dim={index.d}  ntotal={index.ntotal}")

    # 6. Persist
    save_index(index, chunks, collection)

    return index, chunks


def load_index(collection: str) -> tuple[faiss.IndexFlatIP, list[dict]]:
    """
    Load a previously built index and metadata from data/processed/.

    Returns:
        (index, meta_list) ready for search.
    """
    index_path = PROCESSED_DIR / f"{collection}_index.faiss"
    meta_path  = PROCESSED_DIR / f"{collection}_index_meta.jsonl"

    if not index_path.exists():
        raise FileNotFoundError(
            f"{index_path} not found. Run build_index('{collection}') first."
        )

    index = faiss.read_index(str(index_path))
    meta  = [
        json.loads(line)
        for line in meta_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print(f"  Loaded index: dim={index.d}  ntotal={index.ntotal}")
    return index, meta


def search(
    query: str,
    index: faiss.IndexFlatIP,
    meta: list[dict],
    model: SentenceTransformer,
    top_k: int = 5,
) -> list[dict]:
    """
    Encode a query and return the top-k most similar chunks.

    bge-m3 asymmetric retrieval:
      - Passages are stored WITHOUT a prefix (done at index-build time).
      - Queries must be prefixed with QUERY_PREFIX so the model maps them
        into the same embedding space as the passage vectors.
      - This asymmetry is specific to bge-m3;

    Returns a list of metadata dicts each with an added 'score' key
    (cosine similarity; higher = more relevant).
    """
    prefixed_query = QUERY_PREFIX + query

    q_emb = model.encode(
        [prefixed_query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    scores, indices = index.search(q_emb, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:   # FAISS sentinel for empty result slots
            continue
        record = dict(meta[idx])
        record["score"] = float(score)
        results.append(record)

    return results