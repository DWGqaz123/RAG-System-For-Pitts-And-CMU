import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GENERATOR_MODEL   = "Qwen/Qwen2.5-14B-Instruct:featherless-ai"
TOP_K_RETRIEVAL   = 20
TOP_N_CONTEXT     = 10
MAX_CONTEXT_CHARS = 1000
MAX_NEW_TOKENS    = 512
TEMPERATURE       = 0.1

COLLECTIONS = ["A", "B", "C", "D"]

GENERATION_SYSTEM_PROMPT = """\
You are the final answer generator in a Retrieval-Augmented Generation (RAG) system.

Your objective:
Provide the direct and helpful answer using the retrieved context.

IMPORTANT PRIORITY:
1) Try your best to answer the question directly.
2) Use the retrieved context as the primary source.
3) Do NOT say "I don't know" unless the context is completely unrelated.
4) If multiple documents conflict, prefer:
   - More specific information over general information
   - More recent information over older information

Grounding Rules:
- Do NOT invent specific numbers, dates, policies, or names not found in the context.
- If exact details are missing, provide a general but relevant explanation based on what is available.

For event-related questions:
- Only consider recurring events OR events on/after March 19, 2026.
- Ignore past non-recurring events before that date.

Answer style:
- Be clear and concise.
- Answer the question concisely. 
- Do NOT mention "retrieved context" or "documents".
- Do NOT fabricate missing details.
- Do not repeat the question. Use the same terminology as the context
"""


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class RAGResult:
    query:       str
    collection:  list[str]        
    answer:      str
    context:     list[dict]
    latency_ms:  dict[str, float] = field(default_factory=dict)

    def pretty(self) -> str:
        lines = [
            f"Query      : {self.query}",
            f"Collection : {self.collection}",
            "",
            f"Answer\n------\n{self.answer}",
            "",
            f"Context ({len(self.context)} chunks used)",
            "-" * 50,
        ]
        for i, c in enumerate(self.context, 1):
            sec  = c.get("section") or c.get("md_title") or ""
            text = c.get("_text_preview", "")
            lines.append(
                f"  [{i}] rank={c['rerank_rank']}  score={c['rerank_score']:.4f}"
                f"  {c['chunk_id']}\n"
                f"      section : {sec[:1000]}\n"
                f"      preview : {text[:1000]}..."
            )
        lines.append("")
        lat = self.latency_ms
        lines.append(
            f"Latency  route={lat.get('route',0):.0f}ms  "
            f"dense={lat.get('dense',0):.0f}ms  "
            f"sparse={lat.get('sparse',0):.0f}ms  "
            f"rerank={lat.get('rerank',0):.0f}ms  "
            f"generate={lat.get('generate',0):.0f}ms  "
            f"total={lat.get('total',0):.0f}ms"
        )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# RAGPipeline
# ---------------------------------------------------------------------------

class RAGPipeline:

    def __init__(
        self,
        collections:     list[str] = COLLECTIONS,
        top_k_retrieval: int = TOP_K_RETRIEVAL,
        top_n_rerank:    int = 2*TOP_K_RETRIEVAL,
        top_n_context:   int = TOP_N_CONTEXT,
        generator_model: str = GENERATOR_MODEL,
        env_file:        str | Path = ".env",
        verbose:         bool = True,
    ) -> None:
        self.collections     = collections
        self.top_k_retrieval = top_k_retrieval
        self.top_n_rerank    = top_n_rerank
        self.top_n_context   = top_n_context
        self.generator_model = generator_model
        self.verbose         = verbose

        load_dotenv(dotenv_path=env_file)
        self._token = os.getenv("HF_TOKEN", "").strip()
        if not self._token:
            raise EnvironmentError(
                "HF_TOKEN not found.\n"
            )
        self._hf_client = InferenceClient(token=self._token)

        self._log("Initialising RAG pipeline...")
        self._log("  (local reranking with BAAI/bge-reranker-v2-m3; generation with meta-llama/Llama-3.1-8B-Instruct)")

        from scripts.build_index import load_index
        from scripts.build_bm25  import load_bm25
        from scripts.reranker    import _load_chunk_texts, load_reranker
        from scripts.query_router import QueryRouter

        # 1. Router
        self._router = QueryRouter(env_file=env_file, verbose=False)
        self._log(f"  ✓ QueryRouter  ({self._router.model_id})")

        # 2. local reranker (cross-encoder)
        self._reranker = load_reranker()
        self._log(f"  ✓ Reranker  ({self._reranker.model_name}  on local MPS)")

        # 3. Index files
        self._faiss:       dict[str, Any]        = {}
        self._faiss_meta:  dict[str, list[dict]] = {}
        self._bm25:        dict[str, Any]        = {}
        self._chunk_texts: dict[str, dict]       = {}

        for col in collections:
            self._faiss[col], self._faiss_meta[col] = load_index(col)
            self._bm25[col]        = load_bm25(col)
            self._chunk_texts[col] = _load_chunk_texts(col)
            n = len(self._chunk_texts[col])
            self._log(f"  ✓ Collection {col}  ({n} chunks)")
        
        from scripts.build_index import _load_model as _load_embedder, _detect_device
        self._embedder = _load_embedder(_detect_device())
        self._log(f"  ✓ Embedder  (bge-m3  on {self._embedder.device})")

        self._log("Pipeline ready.\n")

        self._log("Pipeline ready.\n")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, query: str) -> RAGResult:
        from scripts.build_index import search, _load_model as _load_embedder, _detect_device
        from scripts.build_bm25  import bm25_search
        from scripts.reranker    import merge_results, rerank

        t_total = time.time()
        latency: dict[str, float] = {}

        self._log(f"\n{'─'*55}")
        self._log(f"Query: {query!r}")

        # ── 1. Route → top-2 collections ───────────────────────────
        t0 = time.time()
        routed_cols = self._router.route(query)
        routed_cols = self._normalize_collections(routed_cols)
        latency["route"] = (time.time() - t0) * 1000
        self._log(f"[1] Route → {routed_cols}  ({latency['route']:.0f}ms)")

        # ── 2. Dense retrieval ── run on routed collection(s), merge results ─────────────
        t0       = time.time()
        # embedder = _load_embedder(_detect_device())
        # dense_hits: list[dict] = []
        # for col in routed_cols:
        #     hits = search(
        #         query,
        #         self._faiss[col],
        #         self._faiss_meta[col],
        #         embedder,
        #         top_k=self.top_k_retrieval,
        #     )
        #     dense_hits.extend(hits)
        # del embedder    # free ~560 MB
        
        dense_hits: list[dict] = []
        for col in routed_cols:
            hits = search(
                query,
                self._faiss[col],
                self._faiss_meta[col],
                self._embedder,
                top_k=self.top_k_retrieval,
            )
            dense_hits.extend(hits)
        
        latency["dense"] = (time.time() - t0) * 1000
        self._log(f"[2] Dense  → {len(dense_hits)} hits  ({latency['dense']:.0f}ms)")

        # ── 3. Sparse retrieval ─ run on routed collection(s), merge results ─────────────────────────────
        t0 = time.time()
        sparse_hits: list[dict] = []
        for col in routed_cols:
            hits = bm25_search(
                query,
                retriever=self._bm25[col],
                top_k=self.top_k_retrieval,
            )
            sparse_hits.extend(hits)
        latency["sparse"] = (time.time() - t0) * 1000
        self._log(f"[3] Sparse → {len(sparse_hits)} hits  ({latency['sparse']:.0f}ms)")

        # ── 4. Merge + Rerank ─────────────────────────────────────────
        combined_chunk_texts: dict[str, str] = {}
        for col in routed_cols:
            combined_chunk_texts.update(self._chunk_texts[col])
        candidates = merge_results(dense_hits, sparse_hits)
        t0 = time.time()
        reranked = rerank(
            query,
            candidates,
            self._reranker,
            combined_chunk_texts,
            top_n=self.top_n_rerank,
        )
        latency["rerank"] = (time.time() - t0) * 1000
        self._log(
            f"[4] Rerank → {len(candidates)} candidates → "
            f"{len(reranked)} hits  ({latency['rerank']:.0f}ms)"
        )

        # ── 5. Build context ──────────────────────────────────────────
        context_chunks = reranked[: self.top_n_context]
        context_str    = self._build_context(context_chunks, combined_chunk_texts)
        for hit in context_chunks:
            hit["_text_preview"] = combined_chunk_texts.get(hit["chunk_id"], "")[:150]

        # ── 6. Generate ───────────────────────────────────────────────
        t0 = time.time()
        answer = self._generate(query, context_str)
        latency["generate"] = (time.time() - t0) * 1000
        self._log(f"[6] Generate  ({latency['generate']:.0f}ms)")

        latency["total"] = (time.time() - t_total) * 1000
        self._log(f"    Total: {latency['total']:.0f}ms")

        return RAGResult(
            query      = query,
            collection = routed_cols,
            answer     = answer,
            context    = context_chunks,
            latency_ms = latency,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_context(
        self,
        chunks: list[dict],
        chunk_texts: dict[str, str],   # ← 直接传合并后的 texts，不再需要 collection 参数
    ) -> str:
        parts: list[str] = []
        for i, hit in enumerate(chunks, 1):
            text = chunk_texts.get(hit["chunk_id"], "").strip()
            if not text:
                continue
            if len(text) > MAX_CONTEXT_CHARS:
                text = text[:MAX_CONTEXT_CHARS] + "…"
            section = hit.get("section") or hit.get("md_title") or ""
            header  = f"[Source {i}: {hit['chunk_id']}  {section}]"
            parts.append(f"{header}\n{text}")
        return "\n\n".join(parts)

    def _generate(self, query: str, context: str) -> str:
        response = self._hf_client.chat_completion(
            model    = self.generator_model,
            messages = [
                {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
                {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {query}"},
            ],
            max_tokens  = MAX_NEW_TOKENS,
            temperature = TEMPERATURE,
            seed        = 42,
        )
        return response.choices[0].message.content.strip()

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def _normalize_collections(self, cols: list[str]) -> list[str]:
        """Return two allowed collections (unique, ordered); fallback to first two configured."""
        cleaned: list[str] = []
        for col in cols:
            up = str(col).strip().upper()
            if up in self.collections and up not in cleaned:
                cleaned.append(up)
            if len(cleaned) == 2:
                break

        if not cleaned:
            cleaned = self.collections[:2]
        elif len(cleaned) == 1:
            for col in self.collections:
                if col not in cleaned:
                    cleaned.append(col)
                    break
        return cleaned[:2]
