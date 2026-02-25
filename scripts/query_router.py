"""
query_router.py
---------------
LLM-based query router for the Pittsburgh RAG system.

Uses meta-llama/Llama-3.1-8B-Instruct via the HuggingFace Serverless
Inference API (Inference Providers) to classify a user query into one
of four collection labels (A / B / C / D).

How it works
------------
Llama-3.1-8B-Instruct is an instruction-tuned chat model.  We send a
structured chat request (system prompt + user message) and constrain the
output to a single letter by setting max_tokens=5 and temperature=0.0.
The system prompt encodes the routing rules; the model outputs "A", "B",
"C", or "D".  A regex fallback handles any unexpected output.

Collection taxonomy
-------------------
  A — General knowledge: Pittsburgh history, demographics, geography,
      CMU general info, Wikipedia-style background.
  B — Regulatory / fiscal: laws, taxes, budget, government spending,
      permits, regulations, finance.
  C — Live / time-sensitive: events, concerts, festivals, schedules,
      calendars, "what's on this weekend".
  D — Specific local entities: museums, sports teams (Steelers, Pirates,
      Penguins), theaters, restaurants, venues.

Setup
-----
  1. Create a .env file in the project root:
         HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx
  2. pip install huggingface_hub python-dotenv
  3. Make sure your HF account has access to Meta Llama models
     (accept the license at https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)

Usage
-----
    from query_router import QueryRouter
    router = QueryRouter()
    label  = router.route("What is the payroll tax rate in Pittsburgh?")
    # -> "B"
"""

import os
import re
from pathlib import Path

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
Given a user question, select the TWO most likely collections to retrieve from.
Rank them by likelihood (most likely first).

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
- STRICT temporal management:
  Only relevant events are:
    (1) recurring events (weekly/monthly/annual), OR
    (2) events on or after March 19, 2026.

D. Culture & Sports
- Museums, symphony, food festivals, sports teams.
- Mix of static info (what it is, history, venue info) and dynamic info (exhibits, games).
- If mainly asking about WHAT an institution/team is → D.
- If mainly asking about specific upcoming schedules → prefer C.

Decision logic:
1) If about regulations, taxes, compliance, financial numbers → B must be included.
2) If explicitly asking for upcoming events, dates, schedules → C must be included.
3) If about museums, symphony, festivals, sports teams → D is likely.
4) Otherwise default to A.
5) Always return EXACTLY TWO collections.

Output EXACTLY TWO letters separated by a comma, ranked by relevance (most relevant first).
Example: "A,C" or "D,B".
Do not output any explanation, punctuation, or extra text."""


# ---------------------------------------------------------------------------
# Router class
# ---------------------------------------------------------------------------

class QueryRouter:
    """
    Routes a natural-language query to a collection label (A/B/C/D)
    using meta-llama/Llama-3.1-8B-Instruct via the HF Serverless
    Inference API (Inference Providers).

    The model receives a system prompt defining routing rules and outputs
    a single letter.  max_tokens=5 keeps cost minimal; temperature=0.0
    makes results deterministic.
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
            # _call_llm 重试耗尽后仍抛出的 402/429，在这里兜底
            if "402" in err or "429" in err:
                print(f"  [router] API 额度问题，使用 fallback: {err[:50]}")
                labels = [FALLBACK, "D"]
            else:
                print(f"  [router] API error: {exc}")
                labels = [FALLBACK, "D"]

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
        Send one chat-completion request and parse the single-letter response.

        max_tokens=5  — we only need one letter; limits cost and latency.
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
                        return [FALLBACK, "D"]
                else:
                    raise   # 非 402/429 错误直接抛出

        return [FALLBACK, "D"]

    @staticmethod
    def _parse_label(raw: str) -> list[str]:
        cleaned = raw.strip().upper()

        # find all standalone letters A/B/C/D in the output, preserving order but removing duplicates
        matches = list(dict.fromkeys(re.findall(r"\b([ABCD])\b", cleaned)))

        if len(matches) >= 2:
            return matches[:2]
        if len(matches) == 1:
            # if only one label is found, pad with a fallback to ensure we always return two labels
            fallback = next((x for x in VALID_LABELS if x != matches[0]), FALLBACK)
            print(f"  [router] only one label parsed from {raw!r} — padding with '{fallback}'")
            return [matches[0], fallback]

        print(f"  [router] unparseable output: {raw!r} — using fallback")
        return [FALLBACK, "B"]

    @staticmethod
    def _ensure_two_labels(labels: list[str]) -> list[str]:
        cleaned: list[str] = []
        for label in labels:
            up = str(label).strip().upper()
            if up in VALID_LABELS and up not in cleaned:
                cleaned.append(up)

        if len(cleaned) >= 2:
            return cleaned[:2]
        if len(cleaned) == 1:
            second = next((x for x in sorted(VALID_LABELS) if x != cleaned[0]), "D")
            return [cleaned[0], second]
        return [FALLBACK, "D"]


# ---------------------------------------------------------------------------
# CLI quick-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    router = QueryRouter()

    test_queries = [
        ("What is the payroll tax rate in Pittsburgh?",   "B"),
        ("Who founded Carnegie Mellon University?",       "A"),
        ("Are there any concerts next Friday downtown?",  "C"),
        ("Tell me about the Steelers' new stadium.",      "D"),
        ("What is the population of Pittsburgh?",         "A"),
        ("What permits do I need to open a restaurant?",  "B"),
        ("What exhibitions are at Carnegie Museum?",      "D"),
        ("When is the next Pittsburgh Symphony concert?", "C"),
    ]

    print("\n" + "=" * 60)
    print("QueryRouter — self-test")
    print("=" * 60)
    correct = 0
    for query, expected in test_queries:
        label = router.route(query)
        mark  = "✓" if label == expected else "✗"
        print(f"  {mark}  [{label}] expected=[{expected}]  {query}")
        correct += label == expected
    print(f"\n  Accuracy: {correct}/{len(test_queries)}")
