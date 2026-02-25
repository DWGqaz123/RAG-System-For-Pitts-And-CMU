"""
chunk_collection_b.py
---------------------
Chunking pipeline for Collection B (Regulations & Finance).

Two documents, two strategies:

  1. payroll_tax_regulations.md
     → Standard heading-aware chunking (reuses chunker.py logic).
       The tax regulations are prose + numbered sections, well suited
       to the H1/H2/H3 hierarchy parser.

  2. 2025_operating_budget.md
     → Table-aware budget chunker.
       The budget is almost entirely financial tables converted from PDF.
       Each row becomes one self-contained chunk that carries full context:
       the section heading, table column headers, item description, and
       dollar-value sequence.  This enables exact retrieval of individual
       line items ("What is the 2025 Police Bureau overtime budget?").

Both chunkers produce records with the same schema so they can be merged
into a single B_chunks.jsonl.

Output:
    data/processed/B_chunks.jsonl

Usage:
    from chunk_collection_b import chunk_collection_b
    chunks = chunk_collection_b()
"""

import json
import re
from pathlib import Path


# Config

PROCESSED_DIR = Path("data/processed")

DOC_TAX = {
    "doc_id":     "B_payroll_tax_regulations",
    "collection": "B",
    "source":     "pittsburghpa.gov",
    "url":        "https://www.pittsburghpa.gov/files/assets/city/v/1/finance/documents/"
                  "tax-forms/9626_payroll_tax_regulations.pdf",
    "title":      "City of Pittsburgh Payroll Tax Regulations",
}

DOC_BUDGET = {
    "doc_id":     "B_2025_operating_budget",
    "collection": "B",
    "source":     "pittsburghpa.gov",
    "url":        "https://www.pittsburghpa.gov/files/assets/city/v/4/omb/documents/"
                  "operating-budgets/2025-operating-budget.pdf",
    "title":      "City of Pittsburgh 2025 Operating Budget",
}


# ──────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────

_YAML_BLOCK = re.compile(r'^---\n.*?\n---\n', re.DOTALL)

def _strip_yaml(text: str) -> str:
    """Remove YAML front-matter block from a processed .md file."""
    return _YAML_BLOCK.sub('', text).strip()


def _load_doc_text(filename: str) -> str:
    path = PROCESSED_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run process_all() from scrape_pdf_collection_b.py first."
        )
    return _strip_yaml(path.read_text(encoding="utf-8"))


# ──────────────────────────────────────────────
# Strategy 1 for Tax Regulations — heading-aware
# (delegates to chunker.py, same logic as Collection A)

def chunk_tax_regulations() -> list[dict]:
    """
    Chunk the payroll tax regulations using the standard heading-aware
    chunker from chunker.py.  Returns a list of chunk dicts.
    """
    # Import lazily so this file is usable even if chunker isn't on the path
    try:
        from scripts.chunkers.chunker_collection_a import build_chunks
    except ImportError:
        raise ImportError("chunker.py not found. Ensure it is in the same directory.")

    text = _load_doc_text("payroll_tax_regulations.md")
    doc  = {**DOC_TAX, "text": text}
    chunks = build_chunks(doc)
    print(f"  Tax regulations : {len(chunks)} chunks (heading-aware)")
    return chunks


# Strategy 2 for Operating Budget — table-aware

# Regex patterns
_HAS_DOLLAR   = re.compile(r'\$\s*[\d,]+')
_YEAR_HEADER  = re.compile(r'\b(20\d{2})\b.*\bTotal\b', re.IGNORECASE)
_IS_SEPARATOR = re.compile(r'^\s*[-|:=]{3,}\s*$')
_BOLD         = re.compile(r'\*{1,2}([^*]+)\*{1,2}')
_HEADING_MD   = re.compile(r'^(#{1,6})\s+(.*)')


def _clean_line(line: str) -> str:
    """Strip Markdown bold/italic markers and excess whitespace."""
    return _BOLD.sub(r'\1', line).strip()


def _is_heading(line: str) -> tuple[int, str] | None:
    """Return (level, text) if the line is a Markdown heading, else None."""
    m = _HEADING_MD.match(line.strip())
    if m:
        return len(m.group(1)), m.group(2).strip()
    return None


def _extract_amounts(parts: list[str]) -> list[str]:
    """Reconstruct dollar-prefixed amount strings from split parts."""
    return ["$" + p.strip() for p in parts if p.strip()]


def chunk_budget_tables(text: str, doc_meta: dict) -> list[dict]:
    """
    Table-aware chunker for the 2025 Operating Budget.

    Parsing state machine:
      - Tracks the current heading hierarchy (H1 → H2 → H3) as 'context'.
      - Tracks the most recently seen table column header line.
      - For every data row (line containing "$<digits>"):
          • Splits on "$" to separate item description from values.
          • Emits one chunk per row, injecting full context + headers.
      - Non-data, non-header lines accumulate as narrative context
        (resolutions, fund descriptions, notes).

    Each chunk text format:
        Title: ...
        Section: ...
        Context: <breadcrumb of headings and narrative>
        Table Headers: <column header row>
        Item: <row description>
        Values: <$x | $y | $z ...>

    This format is highly retrievable: a question like
    "What is the 2025 Police Bureau salary budget?" will embed close
    to the chunk that contains "Police Bureau" + "Salaries" + "$<amount>".
    """
    doc_id     = doc_meta["doc_id"]
    collection = doc_meta["collection"]
    source     = doc_meta["source"]
    url        = doc_meta["url"]
    title      = doc_meta["title"]

    lines = text.splitlines()
    chunks: list[dict] = []

    # Heading breadcrumb: {level: heading_text}
    heading_ctx: dict[int, str] = {}
    # Ordered narrative lines accumulated since last heading/header change
    narrative: list[str] = []
    # Most recently seen table column header
    current_headers: str = ""
    # Section label for chunk metadata (top heading in current context)
    current_section: str = title

    def _breadcrumb() -> str:
        """Build a readable breadcrumb from active headings."""
        parts = [heading_ctx[lvl] for lvl in sorted(heading_ctx) if lvl in heading_ctx]
        return " > ".join(parts) if parts else title

    def _emit_row(item_desc: str, amounts: list[str]):
        """Assemble and append one budget row chunk."""
        narrative_str = " | ".join(narrative) if narrative else ""
        context_str   = _breadcrumb()
        if narrative_str:
            context_str += f" | {narrative_str}"

        chunk_text = "\n".join(filter(None, [
            f"Title: {title}",
            f"Section: {current_section}",
            f"Context: {context_str}",
            f"Table Headers: {current_headers}" if current_headers else None,
            f"Item: {item_desc}",
            f"Values: {' | '.join(amounts)}",
        ]))

        idx = len(chunks)
        chunks.append({
            "chunk_id":    f"{doc_id}__{idx:04d}",
            "doc_id":      doc_id,
            "collection":  collection,
            "source":      source,
            "url":         url,
            "title":       title,
            "md_title":    title,
            "section":     current_section,
            "subsection":  heading_ctx.get(3) or heading_ctx.get(2),
            "chunk_index": idx,
            "text":        chunk_text,
        })

    for raw_line in lines:
        line = _clean_line(raw_line)

        if not line or _IS_SEPARATOR.match(line):
            continue

        # ── Heading line ──────────────────────────────────────────
        h = _is_heading(raw_line)
        if h:
            level, heading_text = h
            # Clear deeper headings when a shallower one appears
            heading_ctx = {k: v for k, v in heading_ctx.items() if k < level}
            heading_ctx[level] = heading_text
            # Update section label to the shallowest heading
            current_section = heading_ctx.get(1) or heading_ctx.get(2) or title
            # New heading context = fresh narrative slate
            narrative = []
            current_headers = ""
            continue

        # ── Table column header row ───────────────────────────────
        if _YEAR_HEADER.search(line):
            current_headers = line
            continue

        # ── Data row (contains dollar amounts) ───────────────────
        if _HAS_DOLLAR.search(line):
            # Special case: "Expected Cash Flow" type global context lines
            if re.search(r'expected cash flow|fund balance|beginning balance',
                         line, re.IGNORECASE):
                narrative.append(line)
                continue

            parts = line.split('$')
            item_desc = parts[0].strip(' |')    # strip table pipes too
            amounts   = _extract_amounts(parts[1:])

            if item_desc or amounts:
                _emit_row(item_desc, amounts)
            continue

        # ── Narrative / context line ──────────────────────────────
        # Only keep lines with meaningful content (skip lone pipes, short noise)
        stripped = line.strip('| \t')
        if len(stripped) > 10:
            narrative.append(stripped)
            # Keep narrative window bounded — older lines matter less
            if len(narrative) > 6:
                narrative = narrative[-6:]

    print(f"  Operating budget: {len(chunks)} chunks (table-aware)")
    return chunks


def chunk_operating_budget() -> list[dict]:
    text = _load_doc_text("2025_operating_budget.md")
    return chunk_budget_tables(text, DOC_BUDGET)


# Combined pipeline

def chunk_collection_b(output_path: Path | None = None) -> list[dict]:
    """
    Chunk both Collection B documents and write B_chunks.jsonl.

    Returns the combined list of all chunk dicts.
    """
    output_path = output_path or PROCESSED_DIR / "B_chunks.jsonl"

    print("=" * 60)
    print("Chunking Collection B")
    print("=" * 60)

    tax_chunks    = chunk_tax_regulations()
    budget_chunks = chunk_operating_budget()
    all_chunks    = tax_chunks + budget_chunks

    # Re-index chunk_index globally across both docs
    # (each doc's internal numbering is already 0-based; keep doc-scoped IDs)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    size_kb = output_path.stat().st_size / 1024
    print(f"\n  → Wrote {len(all_chunks)} total chunks to {output_path}  ({size_kb:.1f} KB)")
    print(f"     tax regulations : {len(tax_chunks)}")
    print(f"     operating budget: {len(budget_chunks)}")

    return all_chunks