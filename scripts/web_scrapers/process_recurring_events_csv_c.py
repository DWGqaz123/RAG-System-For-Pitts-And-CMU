"""
process_recurring_events_csv.py
--------------------------------
Convert the Pittsburgh recurring events CSV into:
  1. data/processed/recurring_events_pittsburgh.md   (cleaned Markdown)
  2. data/processed/C_docs.jsonl                     (appended with this doc)

CSV columns:
  fdn-teaser-headline          → event name / title
  fdn-teaser-subheadline       → schedule (e.g. "First Thursday of every month, 5:30 p.m.")
  fdn-event-teaser-location-link → venue name
  fdn-inline-split-list        → street address
  uk-text-muted                → neighborhood / city
  fdn-event-teaser-price       → price (empty = free or not listed)
  fdn-teaser-tag-link          → category tag
  fdn-teaser-description       → full description text

Usage (import in notebook):
    from process_recurring_events_csv import process_csv
    result = process_csv("data/raw/recurring_events.csv")
"""

import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path


# Section: Configuration.

PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

DOC_ID    = "C_recurring_events_pittsburgh"
SOURCE    = "pittsburghmagazine.com"   # typical source for this dataset
OUT_MD    = PROCESSED_DIR / "recurring_events_pittsburgh.md"
OUT_JSONL = PROCESSED_DIR / "C_docs.jsonl"

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


# Section: Cleaning helpers.

_PHONE_RE    = re.compile(r'\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b')
_WHITESPACE  = re.compile(r'\s{2,}')
_BLANK_LINES = re.compile(r'\n{3,}')


def _clean_field(text: str) -> str:
    """Strip phone numbers, excess whitespace, and normalise newlines."""
    text = _PHONE_RE.sub('', text)
    text = text.replace('\r', ' ').strip()
    # Collapse repeated spaces while preserving line breaks.
    lines = [_WHITESPACE.sub(' ', ln).strip() for ln in text.splitlines()]
    text  = '\n'.join(ln for ln in lines if ln)
    return _BLANK_LINES.sub('\n', text).strip()


def _price_label(price_raw: str) -> str:
    p = price_raw.strip()
    if not p:
        return "Not listed"
    try:
        val = float(p.replace('$', '').replace(',', ''))
        return "Free" if val == 0 else f"${val:.2f}"
    except ValueError:
        return p


# Section: CSV loading.

def load_csv(csv_path: str | Path) -> list[dict]:
    """
    Read the CSV and return a list of cleaned record dicts.
    Handles the messy multi-line description field gracefully.
    """
    records = []
    with open(csv_path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rec = {
                "title":    _clean_field(row.get(COLUMNS["title"],    "")),
                "schedule": _clean_field(row.get(COLUMNS["schedule"], "")),
                "venue":    _clean_field(row.get(COLUMNS["venue"],    "")),
                "address":  _clean_field(row.get(COLUMNS["address"],  "")),
                "area":     _clean_field(row.get(COLUMNS["area"],     "")),
                "price":    _price_label(row.get(COLUMNS["price"],    "")),
                "category": _clean_field(row.get(COLUMNS["category"], "")),
                "desc":     _clean_field(row.get(COLUMNS["desc"],     "")),
            }
            if rec["title"]:   # skip empty rows
                records.append(rec)
    return records


# Section: Markdown rendering.

def _record_to_md(rec: dict) -> str:
    """Render one event record as a Markdown section."""
    lines = [f"## {rec['title']}"]
    if rec["schedule"]:
        lines.append(f"**Schedule:** {rec['schedule']}")
    if rec["venue"]:
        lines.append(f"**Venue:** {rec['venue']}")
    if rec["address"]:
        lines.append(f"**Address:** {rec['address']}")
    if rec["area"]:
        lines.append(f"**Area:** {rec['area']}")
    lines.append(f"**Price:** {rec['price']}")
    if rec["category"]:
        lines.append(f"**Category:** {rec['category']}")
    if rec["desc"]:
        lines.append("")
        lines.append(rec["desc"])
    return "\n".join(lines)


def records_to_markdown(records: list[dict], source_url: str = "") -> str:
    """Combine all records into a single Markdown document."""
    header = (
        f"---\n"
        f"source_url: {source_url}\n"
        f"scraped_at: {datetime.utcnow().isoformat()}Z\n"
        f"title: Pittsburgh Recurring Events\n"
        f"source: {SOURCE}\n"
        f"group: recurring_events\n"
        f"---\n\n"
        f"# Pittsburgh Recurring Events\n\n"
        f"A curated list of recurring events in the Pittsburgh area.\n"
    )
    sections = "\n\n".join(_record_to_md(r) for r in records)
    return header + "\n\n" + sections


# Section: JSONL record builder.

def _build_doc_record(records: list[dict], source_url: str) -> dict:
    """
    Build the C_docs.jsonl record for the whole recurring-events document.
    The 'text' field is the full cleaned Markdown body (without YAML header).
    """
    body_lines = []
    for rec in records:
        body_lines.append(_record_to_md(rec))
    return {
        "doc_id":     DOC_ID,
        "collection": "C",
        "source":     SOURCE,
        "url":        source_url,
        "title":      "Pittsburgh Recurring Events",
        "fetched_at": datetime.utcnow().isoformat() + "Z",
        "text":       "\n\n".join(body_lines),
    }


def _append_to_jsonl(record: dict, path: Path):
    """Append one JSON record to a JSONL file (creates file if absent)."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# Section: Main pipeline.

def process_csv(csv_path: str | Path,
                source_url: str = "",
                append_jsonl: bool = True) -> dict:
    """
    Full pipeline: CSV → cleaned Markdown + C_docs.jsonl entry.

    Args:
        csv_path:      Path to the input CSV file.
        source_url:    Original URL for the data (used in metadata).
        append_jsonl:  If True, append to C_docs.jsonl (default).
                       Set False to skip JSONL output.

    Returns:
        A result dict with paths and counts.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    print(f"  ↓  Reading: {csv_path.name}")
    records = load_csv(csv_path)
    print(f"  ✓  Loaded {len(records)} records")

    md_text = records_to_markdown(records, source_url=source_url)
    OUT_MD.write_text(md_text, encoding="utf-8")
    print(f"  ✓  Markdown → {OUT_MD}  ({OUT_MD.stat().st_size / 1024:.1f} KB)")

    if append_jsonl:
        doc_record = _build_doc_record(records, source_url=source_url)
        _append_to_jsonl(doc_record, OUT_JSONL)
        print(f"  ✓  Appended doc record to {OUT_JSONL}")

    return {
        "records":    len(records),
        "md_path":    str(OUT_MD),
        "jsonl_path": str(OUT_JSONL) if append_jsonl else None,
    }
