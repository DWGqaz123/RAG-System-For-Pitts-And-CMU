"""
chunk_recurring_events.py
--------------------------
Chunk the Pittsburgh recurring events Markdown file into C_chunks.jsonl.

One record = one chunk. Each record carries:
  - Structured 'text' for embedding (title, schedule, venue, area, category, description)
  - Metadata fields for hard-filtering: category, area, price_raw, is_free

The recurring events dataset is fundamentally different from calendar events:
  • No specific date — events repeat on a fixed schedule ("First Thursday of
    every month"), so there is no iso_dt to filter on.
  • Rich description text → embedding-friendly chunk.
  • Category and area tags → useful for metadata filtering.

Output
------
    data/processed/C/C_docs.jsonl    (appended)
    data/processed/C/C_chunks.jsonl   (appended)
    from chunk_recurring_events import chunk_recurring_events
    chunks = chunk_recurring_events("data/raw/C/recurring_events.csv")
"""

import csv
import json
import re
from pathlib import Path


# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

PROCESSED_DIR = Path("data/processed/C")
OUT_CHUNKS    = PROCESSED_DIR / "C_chunks.jsonl"

DOC_ID  = "C_recurring_events_pittsburgh"
SOURCE  = "pittsburghmagazine.com"
URL     = ""   # set at runtime via chunk_recurring_events(source_url=...)

COLUMNS = {
    "title":    "fdn-teaser-headline",
    "schedule": "fdn-teaser-subheadline",
    "venue":    "fdn-event-teaser-location-link",
    "address":  "fdn-inline-split-list",
    "area":     "uk-text-muted",
    "price":    "fdn-event-teaser-price",
    "category": "fdn-teaser-tag-link",
    "desc":     "fdn-teaser-description",
}

# ──────────────────────────────────────────────
# Cleaning (same helpers as process_recurring_events_csv.py)
# ──────────────────────────────────────────────

_PHONE_RE   = re.compile(r'\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b')
_WS         = re.compile(r'\s{2,}')
_BLANK      = re.compile(r'\n{3,}')


def _clean(text: str) -> str:
    text = _PHONE_RE.sub('', text)
    text = text.replace('\r', ' ').strip()
    lines = [_WS.sub(' ', ln).strip() for ln in text.splitlines()]
    return _BLANK.sub('\n', '\n'.join(ln for ln in lines if ln)).strip()


def _price_label(raw: str) -> str:
    p = raw.strip()
    if not p:
        return ""
    try:
        val = float(p.replace('$', '').replace(',', ''))
        return "Free" if val == 0 else f"${val:.2f}"
    except ValueError:
        return p


def _is_free(raw: str) -> bool:
    p = raw.strip().lower()
    if not p:
        return False
    if p == "free":
        return True
    try:
        return float(p.replace('$', '').replace(',', '')) == 0
    except ValueError:
        return False


# ──────────────────────────────────────────────
# Load CSV
# ──────────────────────────────────────────────

def _load_csv(csv_path: Path) -> list[dict]:
    records = []
    with open(csv_path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rec = {
                "title":    _clean(row.get(COLUMNS["title"],    "")),
                "schedule": _clean(row.get(COLUMNS["schedule"], "")),
                "venue":    _clean(row.get(COLUMNS["venue"],    "")),
                "address":  _clean(row.get(COLUMNS["address"],  "")),
                "area":     _clean(row.get(COLUMNS["area"],     "")),
                "price_raw":row.get(COLUMNS["price"], "").strip(),
                "price":    _price_label(row.get(COLUMNS["price"], "")),
                "category": _clean(row.get(COLUMNS["category"], "")),
                "desc":     _clean(row.get(COLUMNS["desc"],     "")),
            }
            if rec["title"]:
                records.append(rec)
    return records


# ──────────────────────────────────────────────
# Build one chunk per record
# ──────────────────────────────────────────────

def _build_chunk(rec: dict, idx: int, source_url: str) -> dict:
    """
    Serialise one recurring event record into a RAG chunk.

    Text format optimised for embedding + LLM readability:

        Event: <title>
        Schedule: <schedule>
        Venue: <venue>
        Address: <address>, <area>
        Price: <price>
        Category: <category>
        Description: <description>

    Metadata fields for hard pre-filtering (no embedding needed):
        category  → filter by type ("Dance", "Music", "Other Stuff" …)
        area      → filter by neighborhood ("Oakland", "Squirrel Hill" …)
        is_free   → filter for free events
    """
    # Build address string
    addr_parts = [p for p in (rec["address"], rec["area"]) if p]
    addr_str   = ", ".join(addr_parts)

    # Embedding text — structured key-value, dense but readable
    lines = [f"Event: {rec['title']}"]
    if rec["schedule"]:
        lines.append(f"Schedule: {rec['schedule']}")
    if rec["venue"]:
        lines.append(f"Venue: {rec['venue']}")
    if addr_str:
        lines.append(f"Address: {addr_str}")
    if rec["price"]:
        lines.append(f"Price: {rec['price']}")
    if rec["category"]:
        lines.append(f"Category: {rec['category']}")
    if rec["desc"]:
        lines.append(f"Description: {rec['desc']}")

    text = "\n".join(lines)

    return {
        # ── Standard collection fields ──────────────────────────
        "chunk_id":    f"{DOC_ID}__{idx:04d}",
        "doc_id":      DOC_ID,
        "collection":  "C",
        "source":      SOURCE,
        "url":         source_url,
        "title":       "Pittsburgh Recurring Events",
        "md_title":    "Pittsburgh Recurring Events",
        "section":     rec["category"] or "Uncategorized",
        "subsection":  rec["area"] or None,
        "chunk_index": idx,
        # ── Event-specific metadata (for hard-filtering) ────────
        "doc_type":    "recurring_event",
        "event_date":  None,        # recurring events have no fixed date
        "schedule":    rec["schedule"],
        "venue":       rec["venue"],
        "area":        rec["area"],
        "category":    rec["category"],
        "is_free":     _is_free(rec["price_raw"]),
        # ── Embedding text ──────────────────────────────────────
        "text":        text,
    }


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

def chunk_recurring_events(csv_path: str | Path,
                            source_url: str = "",
                            output_path: Path | None = None,
                            append: bool = True) -> list[dict]:
    """
    Chunk the recurring events CSV and write to C_chunks.jsonl.

    Args:
        csv_path:     Path to the input CSV file.
        source_url:   Original data URL (stored in metadata).
        output_path:  Override default output path.
        append:       Append to existing file (default True, to merge with
                      other Collection C chunks).

    Returns:
        List of all chunk dicts.
    """
    output_path = output_path or OUT_CHUNKS
    csv_path    = Path(csv_path)

    print("=" * 60)
    print("Chunking — Pittsburgh Recurring Events")
    print("=" * 60)

    records = _load_csv(csv_path)
    print(f"  ✓  Loaded {len(records)} records from {csv_path.name}")

    chunks = [_build_chunk(rec, i, source_url) for i, rec in enumerate(records)]

    mode = "a" if append else "w"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, mode, encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    size_kb = output_path.stat().st_size / 1024
    free_n  = sum(1 for c in chunks if c["is_free"])
    cats    = {}
    for c in chunks:
        cats[c["category"]] = cats.get(c["category"], 0) + 1

    print(f"\n  → {'Appended' if append else 'Wrote'} {len(chunks)} chunks "
          f"to {output_path}  ({size_kb:.1f} KB)")
    print(f"     Free events : {free_n}")
    print(f"     Categories  :")
    for cat, n in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"       {cat or '(none)':<30} {n:>4}")

    return chunks