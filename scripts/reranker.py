"""
reranker.py
-----------
Merge dense (FAISS) and sparse (BM25) retrieval results, deduplicate,
then rerank with a local cross-encoder (BAAI/bge-reranker-v2-m3).

Pipeline (summary):
  dense_results + sparse_results
    → merge & deduplicate on chunk_id
    → look up 'text' from {collection}_chunks.jsonl
    → score with cross-encoder (batched, ≤32 pairs)
    → sigmoid(logits) → relevance scores in [0, 1]
    → return top-n by rerank score
"""

import json
import math
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ---------------------------------------------------------------------------
# Paths & defaults
# ---------------------------------------------------------------------------

PROCESSED_DIR     = Path("data/processed")
RERANKER_MODEL    = "BAAI/bge-reranker-v2-m3"
TOP_K_RETRIEVAL   = 20
TOP_N_RERANK      = 10
API_BATCH_SIZE    = 32    # pairs per batch
MAX_PASSAGE_CHARS = 1500  # truncate passages before scoring


# ---------------------------------------------------------------------------
# Sigmoid
# ---------------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


# ---------------------------------------------------------------------------
# Reranker client (stateless — just holds token + URL)
# ---------------------------------------------------------------------------

class HFReranker:
    """Local cross-encoder using BAAI/bge-reranker-v2-m3.
    Chooses device priority: MPS (Apple Silicon) > CUDA > CPU."""

    def __init__(
        self,
        model_name: str = RERANKER_MODEL,
        device: str | None = None,
        use_fp16: bool = False,
    ) -> None:
        self.model_name = model_name

        # Device selection
        if device:
            self.device = torch.device(device)
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        print(f"  Device          : {self.device}")

        dtype = torch.float16 if use_fp16 else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=dtype,
        ).to(self.device).eval()

    def compute_scores(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Score (query, passage) pairs, returning sigmoid-normalized [0,1] scores.
        Processed in batches of API_BATCH_SIZE."""
        all_scores: list[float] = []
        for i in range(0, len(pairs), API_BATCH_SIZE):
            batch = pairs[i : i + API_BATCH_SIZE]
            all_scores.extend(self._score_batch(batch))
        return all_scores

    def _score_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits.squeeze(-1)

        # MPS/CUDA tensor → CPU → numpy → python float
        logits_cpu = logits.float().cpu().tolist()
        if isinstance(logits_cpu, float):          # batch_size=1 case
            logits_cpu = [logits_cpu]
        return [_sigmoid(x) for x in logits_cpu]

# ---------------------------------------------------------------------------
# Convenience loader (mirrors local version's interface)
# ---------------------------------------------------------------------------

def load_reranker(
    model_name: str = RERANKER_MODEL,
    device: str | None = None,
    use_fp16: bool = False,
    token: str | None = None,   # kept for API parity; not used locally
) -> HFReranker:
    print(f"  Reranker        : {model_name}  (local inference)")
    return HFReranker(model_name=model_name, device=device, use_fp16=use_fp16)


# ---------------------------------------------------------------------------
# Chunk text lookup
# ---------------------------------------------------------------------------

_CHUNK_TEXT_CACHE: dict[str, dict[str, str]] = {}


def _load_chunk_texts(collection: str) -> dict[str, str]:
    """Build chunk_id → text mapping from {collection}_chunks.jsonl (cached)."""
    if collection in _CHUNK_TEXT_CACHE:
        return _CHUNK_TEXT_CACHE[collection]

    path = PROCESSED_DIR / f"{collection}_chunks.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run the chunking pipeline first."
        )

    mapping: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        chunk = json.loads(line)
        mapping[chunk["chunk_id"]] = chunk.get("text", "")

    _CHUNK_TEXT_CACHE[collection] = mapping
    return mapping


# ---------------------------------------------------------------------------
# Merge & deduplicate retrieval results
# ---------------------------------------------------------------------------

def merge_results(
    dense_hits:  list[dict],
    sparse_hits: list[dict],
) -> list[dict]:
    """
    Union dense and sparse results, deduplicating on chunk_id.

    Keeps both original scores as 'dense_score' and 'sparse_score'
    for provenance. Dense hits take precedence for the record base.
    """
    seen: dict[str, dict] = {}

    for hit in dense_hits:
        cid = hit["chunk_id"]
        record = dict(hit)
        record["dense_score"]  = hit["score"]
        record["sparse_score"] = 0.0
        seen[cid] = record

    for hit in sparse_hits:
        cid = hit["chunk_id"]
        if cid in seen:
            seen[cid]["sparse_score"] = hit["score"]
        else:
            record = dict(hit)
            record["dense_score"]  = 0.0
            record["sparse_score"] = hit["score"]
            seen[cid] = record

    return list(seen.values())


# ---------------------------------------------------------------------------
# Core rerank function
# ---------------------------------------------------------------------------

def rerank(
    query:       str,
    candidates:  list[dict],
    reranker:    HFReranker,
    chunk_texts: dict[str, str],
    top_n:       int = TOP_N_RERANK,
) -> list[dict]:
    """
    Score every candidate with the cross-encoder API and return the top-n.

    Passages are truncated to MAX_PASSAGE_CHARS before sending to stay
    within the model's token limit and reduce API payload size.

    Each returned dict gains 'rerank_score' (float in [0,1]) and
    'rerank_rank' (1-indexed int). Candidates with no text get score 0.0.
    """
    # Initialise scores up-front so the field always exists
    for hit in candidates:
        hit["rerank_score"] = 0.0

    if not candidates:
        return []

    # Build (query, passage) pairs for valid candidates only
    pairs:         list[tuple[str, str]] = []
    valid_indices: list[int]             = []

    for i, hit in enumerate(candidates):
        text = chunk_texts.get(hit["chunk_id"], "").strip()
        if text:
            # Truncate long passages to limit API payload & token count
            if len(text) > MAX_PASSAGE_CHARS:
                text = text[:MAX_PASSAGE_CHARS]
            pairs.append((query, text))
            valid_indices.append(i)

    if not pairs:
        ranked = candidates[:top_n]
        for rank, hit in enumerate(ranked, 1):
            hit["rerank_rank"] = rank
        return ranked

    # Call API
    scores = reranker.compute_scores(pairs)

    for list_pos, cand_idx in enumerate(valid_indices):
        candidates[cand_idx]["rerank_score"] = scores[list_pos]

    # Sort descending by reranker score
    ranked = sorted(candidates, key=lambda h: h["rerank_score"], reverse=True)

    for rank, hit in enumerate(ranked[:top_n], start=1):
        hit["rerank_rank"] = rank

    return ranked[:top_n]