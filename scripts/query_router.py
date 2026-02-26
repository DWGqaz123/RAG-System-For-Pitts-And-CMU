"""
query_router.py
---------------
LLM-based query router for the Pittsburgh RAG system.

Given a user question, it returns the two most likely collection labels
(A/B/C/D) using meta-llama/Llama-3.1-8B-Instruct via the HF Inference
API. The system prompt encodes routing rules; a regex fallback guards
against malformed model outputs.
"""

import re
from pathlib import Path
import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_ID     = "meta-llama/Llama-3.1-8B-Instruct"
VALID_LABELS = {"A", "B", "C", "D"}
FALLBACK     = "A"    # returned on API error or unparseable output

SYSTEM_PROMPT = """You are a query router for a RAG system.

Your task:
Given a user question, return the TWO most likely collections to search, ranked by likelihood (most likely first).
Always return exactly two letters (A/B/C/D) separated by a comma. No explanation or extra text.

Collections:

A. General & Historical
- Static, narrative-heavy background information.
- Examples: Wikipedia, Britannica, Visit Pittsburgh general info, CMU About & History.
- Use for general descriptions, history, definitions, overview-type questions.

B. Regulations & Finance
- Highly structured, dense, requires exact retrieval.
- Examples: City Tax Regulations, 2025 Operating Budget.
- Use for rules, eligibility, rates, deadlines, forms, compliance, taxes, budget line items, financial figures.

C. Dynamic Events
- Time-sensitive event listings.
- Examples: Visit PGH events, Downtown events, City Paper events, CMU Campus Calendars.
- Use when user asks for:
    - upcoming events
    - things to do on a specific date
    - this week/weekend
    - schedules or calendars
- STRICT temporal management: Only events that are recurring OR on/after March 19, 2026.

D. Culture & Sports
- Food, restaurants, museums, symphony, festivals, sports teams.
- Mix of static info (what it is, history, venue info) and dynamic info (exhibits, games).
- If mainly asking about WHAT an institution/team is → D.
- If mainly asking about specific upcoming schedules → prefer C.

Decision logic:
1) Regulations/taxes/compliance/financial numbers → include B (often with A for background).
2) Upcoming events/dates/schedules → include C (often with D for venues/teams).
3) Museums/symphony/festivals/sports/food/restaurants → include D; add C if clearly date-bound.
4) Otherwise default to A (may pair with B if rules/numbers appear).
5) Output exactly TWO letters, ordered by likelihood.
"""


# ---------------------------------------------------------------------------
# Router class
# ---------------------------------------------------------------------------

class QueryRouter:
    """
    Routes a natural-language query to two collection labels (A/B/C/D)
    using meta-llama/Llama-3.1-8B-Instruct via the HF Serverless
    Inference API (Inference Providers).

    The model receives a system prompt defining routing rules and outputs
    two letters. max_tokens=5 keeps cost minimal; temperature=0.0 makes
    results deterministic.
    """

    def __init__(
        self,
        model_id: str = MODEL_ID,
        env_file: str | Path = ".env",
        verbose: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        model_id  : HuggingFace model repo id.
        env_file  : Path to the .env file containing HF_TOKEN.
        verbose   : Print routing decisions to stdout.
        """
        load_dotenv(dotenv_path=env_file)
        token = os.getenv("HF_TOKEN", "").strip()
        if not token:
            raise EnvironmentError(
                "HF_TOKEN not found. Add it to your .env file:\n"
                "  HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx"
            )

        self.client   = InferenceClient(token=token)
        self.model_id = model_id
        self.verbose  = verbose

        if verbose:
            print(f"QueryRouter ready — model: {self.model_id}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route(self, query: str) -> list[str]:
        try:
            labels = self._call_llm(query)
        except Exception as exc:
            err = str(exc)
        
            if "402" in err or "429" in err:
                print(f"  [router] use fallback: {err[:50]}")
                labels = [FALLBACK]
            else:
                print(f"  [router] API error: {exc}")
                labels = [FALLBACK]

        labels = self._ensure_two_labels(labels)
        if self.verbose:
            print(f"  [router] '{query}' -> {labels}")
        return labels

    def route_batch(self, queries: list[str]) -> list[list[str]]:
        return [self.route(q) for q in queries]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _call_llm(
        self,
        query: str,
        max_retries: int = 10,
        retry_delay: float = 5.0,
    ) -> list[str]:
        """
        Send one chat-completion request and parse the two-letter response.

        max_tokens=5  — we only need a couple of letters; limits cost and latency.
        temperature=0 — deterministic output.
        seed=42       — reproducible across identical requests.

        Llama-3.1-8B-Instruct uses the standard chat template with
        <|begin_of_text|> / <|eot_id|> tokens internally; the HF
        InferenceClient handles the template automatically.

        Retries up to max_retries times on 402 (Payment Required) and
        429 (Too Many Requests). 429 uses a shorter wait (2s) since it
        typically recovers faster than 402.
        """
        import time

        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.chat_completion(
                    model=self.model_id,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": f"Question: {query}"},
                    ],
                    max_tokens=5,
                    temperature=0.0,
                    seed=42,
                )
                raw = response.choices[0].message.content
                return self._parse_label(raw)

            except Exception as exc:
                err = str(exc)
                if "402" in err or "429" in err:
                    if attempt < max_retries:
                        wait = retry_delay if "402" in err else 2.0
                        print(f"  [router] {err[:50]}，{wait}s 后重试 "
                            f"({attempt}/{max_retries})...")
                        time.sleep(wait)
                    else:
                        print(f"  [router] 重试 {max_retries} 次仍失败，使用 fallback")
                        return [FALLBACK]
                else:
                    raise   

        return [FALLBACK]

    @staticmethod
    def _parse_label(raw: str) -> list[str]:
        cleaned = raw.strip().upper()

        # find all standalone letters A/B/C/D in the output, preserving order but removing duplicates
        matches = list(dict.fromkeys(re.findall(r"\b([ABCD])\b", cleaned)))

        if len(matches) >= 2:
            return matches[:2]
        if len(matches) == 1:
            return matches

        print(f"  [router] unparseable output: {raw!r} — using fallback")
        return [FALLBACK]

    @staticmethod
    def _ensure_two_labels(labels: list[str]) -> list[str]:
        cleaned: list[str] = []
        for label in labels:
            up = str(label).strip().upper()
            if up in VALID_LABELS and up not in cleaned:
                cleaned.append(up)

        if not cleaned:
            return [FALLBACK, "D"]
        if len(cleaned) == 1:
            first = cleaned[0]
            for candidate in ["A", "B", "C", "D"]:
                if candidate != first:
                    cleaned.append(candidate)
                    break
        return cleaned[:2]


# ---------------------------------------------------------------------------
# CLI quick-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    router = QueryRouter()

    test_queries = [
        ("What is the payroll tax rate in Pittsburgh?",   {"B", "A"}),
        ("Who founded Carnegie Mellon University?",       {"A", "D"}),
        ("Are there any concerts next Friday downtown?",  {"C", "D"}),
        ("Tell me about the Steelers' new stadium.",      {"D", "A"}),
        ("What is the population of Pittsburgh?",         {"A", "B"}),
        ("What permits do I need to open a restaurant?",  {"B", "A"}),
        ("What exhibitions are at Carnegie Museum?",      {"D", "A"}),
        ("When is the next Pittsburgh Symphony concert?", {"C", "D"}),
    ]

    print("\n" + "=" * 60)
    print("QueryRouter — self-test")
    print("=" * 60)
    correct = 0
    for query, expected in test_queries:
        labels = router.route(query)
        hit    = labels[0] in expected and labels[1] in expected
        mark   = "✓" if hit else "✗"
        print(f"  {mark}  {labels} expected~{expected}  {query}")
        correct += hit
    print(f"\n  Accuracy (both in top-2 set): {correct}/{len(test_queries)}")
