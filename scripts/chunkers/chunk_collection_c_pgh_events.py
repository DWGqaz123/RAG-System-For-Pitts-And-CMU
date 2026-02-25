"""
chunk_collection_c_pgh_events.py
---------------------------------
Chunking pipeline for Collection C — Pittsburgh Events pages.

Two sources, two parsers, one unified output schema:

  1. pittsburgh.events (pgh_events_<month>.md)
     State-machine parser for vertical event card layout:
       - / month / day / year / time / weekday / title / venue / address

  2. downtownpittsburgh.com (downtown_pgh_events_<NN>.md)
     H1-split parser for the actual scraped format:
       # Event Title
       May 22, 2026 - May 24, 2026 | 12:00 pm - 1:00 pm
       Description text...
       Category,
       Tag,

Both parsers emit chunk dicts with the same schema, including an
'event_date' ISO field for hard date-range pre-filtering.

Output:
    data/processed/C_chunks.jsonl

Usage (import in notebook):
    from chunk_collection_c_pgh_events import chunk_collection_c
    chunks = chunk_collection_c()
"""

import json
import re
from datetime import datetime
from pathlib import Path


# Config

PROCESSED_DIR = Path("data/processed")

PGH_MONTHS = [
    "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
]

DOWNTOWN_MONTHS = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

_MONTH_NUM = {
    "january": "01", "february": "02", "march":     "03",
    "april":   "04", "may":      "05", "june":      "06",
    "july":    "07", "august":   "08", "september": "09",
    "october": "10", "november": "11", "december":  "12",
}

_MONTH_LABEL = {v: k.capitalize() for k, v in _MONTH_NUM.items()}

_YAML_BLOCK = re.compile(r'^---\n.*?\n---\n', re.DOTALL)


# ──────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────

def _strip_yaml(text: str) -> tuple[str, str]:
    """Return (source_url, body) extracted from a processed .md file."""
    url_match = re.search(r'^source_url:\s*(.+)$', text, re.MULTILINE)
    source_url = url_match.group(1).strip() if url_match else ""
    body = _YAML_BLOCK.sub('', text).strip()
    return source_url, body


def _parse_date(month: str, day: str | int, year: str | int,
                time_str: str = "") -> str:
    """
    Return 'YYYY-MM-DD HH:MM' or 'YYYY-MM-DD' on parse failure.
    Handles both 12-hour (7:00 PM) and 24-hour (19:00) time strings.
    """
    try:
        month_num = _MONTH_NUM.get(str(month).lower().strip())
        if not month_num:
            return ""
        date_part = f"{year}-{month_num}-{int(day):02d}"
        if not time_str:
            return date_part
        t = time_str.strip().upper()
        fmt = "%Y-%m-%d %I:%M %p" if ("AM" in t or "PM" in t) else "%Y-%m-%d %H:%M"
        return datetime.strptime(f"{date_part} {t}", fmt).strftime("%Y-%m-%d %H:%M")
    except Exception:
        month_num = _MONTH_NUM.get(str(month).lower().strip(), "00")
        try:
            return f"{year}-{month_num}-{int(day):02d}"
        except Exception:
            return ""


# ──────────────────────────────────────────────
# Source 1: pittsburgh.events parser
# ──────────────────────────────────────────────

_PGH_NOISE = re.compile(
    r'^(Event Info|Filter|Sort By.*|View (All|Calendar|List).*'
    r'|Load More.*|Subscribe.*|Add to Calendar|Share'
    r'|\d+\s+events?|Upcoming Events.*|Tickets?|\|+'
    r'|& Tickets'           # residual "& Tickets" nav fragment
    r'|Events:\s*\d+.*'    # "Events: 255" count line
    r')$',
    re.IGNORECASE,
)


def _pgh_clean_lines(body: str) -> list[str]:
    for anchor in ("## Upcoming Events", "## Events", "# Events"):
        if anchor in body:
            body = body.split(anchor, 1)[1]
            break
    lines = [ln.strip() for ln in body.splitlines()]
    return [ln for ln in lines if ln and not _PGH_NOISE.match(ln)]


def _pgh_parse_events(lines: list[str]) -> list[dict]:
    events, current = [], []

    def _flush(block):
        if len(block) < 8:
            if block:
                print(f"  ⚠  Short card ({len(block)} fields): {block[:3]}")
            return
        month, day, year, time_str, weekday, title, venue = block[:7]
        address = " ".join(block[7:])
        events.append({
            "month": month, "day": day, "year": year,
            "time": time_str, "weekday": weekday,
            "title": title, "venue": venue, "address": address,
            "iso_dt": _parse_date(month, day, year, time_str),
        })

    for line in lines:
        if line == "-":
            _flush(current)
            current = []
        else:
            current.append(line)
    _flush(current)
    return events


def _pgh_build_chunk(ev: dict, month_slug: str,
                     idx: int, global_idx: int, source_url: str) -> dict:
    text = (
        f"Event Name: {ev['title']}\n"
        f"Date: {ev['weekday']}, {ev['month']} {ev['day']}, {ev['year']} at {ev['time']}\n"
        f"Venue: {ev['venue']}\n"
        f"Location: {ev['address']}"
    )
    return {
        "chunk_id":    f"C_pgh_events_{month_slug}__{idx:04d}",
        "doc_id":      f"C_pgh_events_{month_slug}",
        "collection":  "C",
        "source":      "pittsburgh.events",
        "url":         source_url,
        "title":       f"Pittsburgh Events — {month_slug.capitalize()}",
        "md_title":    f"Pittsburgh Events — {month_slug.capitalize()}",
        "section":     f"{ev['month']} {ev['year']}",
        "subsection":  None,
        "chunk_index": global_idx,
        "doc_type":    "event",
        "event_date":  ev["iso_dt"],
        "venue":       ev["venue"],
        "text":        text,
    }


def chunk_pgh_month(month_slug: str, global_offset: int = 0) -> list[dict]:
    path = PROCESSED_DIR / f"pgh_events_{month_slug}.md"
    if not path.exists():
        print(f"  ⚠  Missing: {path.name} — skipping")
        return []
    source_url, body = _strip_yaml(path.read_text(encoding="utf-8"))
    lines  = _pgh_clean_lines(body)
    events = _pgh_parse_events(lines)
    chunks = [
        _pgh_build_chunk(ev, month_slug, i, global_offset + i, source_url)
        for i, ev in enumerate(events)
    ]
    print(f"  ✓  {path.name:<45}  {len(events):>4} events  →  {len(chunks)} chunks")
    return chunks


# ──────────────────────────────────────────────
# Source 2: downtownpittsburgh.com parser
# ──────────────────────────────────────────────
# Strategy: re.split on H1 headings.  Each block contains:
#   • First non-empty line  → date / time range
#   • Remaining lines       → description paragraphs + comma-trailing tags

# Split marker: lines that start with exactly one '#' followed by a space
_DT_H1_SPLIT = re.compile(r'^# (.+)$', re.MULTILINE)

# Date range line patterns:
#   "May 22, 2026 - May 24, 2026 \| 12:00 pm - 1:00 pm"  (full month name)
#   "Mar 14, 2026 \| 11:00 am - 1:00 pm"                 (abbreviated month)
#   "Jan 1, 2026 - Jan 1, 2027 \| 10:00 am - 1:00 pm"    (abbrev range)
_MONTHS_RE = (
    r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?'
    r'|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)'
)
_DT_DATE_LINE = re.compile(
    rf'^({_MONTHS_RE}\s+\d{{1,2}},?\s*\d{{4}})'   # start date (full or abbreviated)
    rf'(?:\s*[-–]\s*{_MONTHS_RE}\s+\d{{1,2}},?\s*\d{{4}})?'  # optional end date
    rf'(?:\s*[|\\\\]\s*(.+))?$',                  # optional time part after | or \|
    re.IGNORECASE,
)

# A tag line: text that ends with a comma, or is a known single-word category.
# Keeps letters, spaces, ampersands, hyphens — no digits or sentence punctuation.
_DT_TAG_LINE = re.compile(r'^[A-Za-z][A-Za-z &\-]+,?$')

# Inline noise to strip from description text
_DT_INLINE_NOISE = re.compile(r'\bREAD MORE\b', re.IGNORECASE)


def _dt_parse_block(title: str, block_text: str) -> dict | None:
    """
    Parse one H1 event block into a structured dict.
    Returns None if no recognisable date line is found.
    """
    date_str  = ""
    time_str  = ""
    iso_dt    = ""
    desc_parts: list[str] = []
    tags:       list[str] = []
    found_date = False

    for raw_line in block_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if not found_date:
            m = _DT_DATE_LINE.match(line)
            if m:
                date_str   = m.group(1).strip()
                time_part  = (m.group(2) or "").strip()
                # Take only the start time (before " - "); strip leading pipe/backslash artefacts
                time_str   = re.sub(r'^[|\\\s]+', '', time_part.split("-")[0]).strip()
                found_date = True
                # Parse ISO date — try full month name first, then abbreviated
                for fmt in ("%B %d, %Y", "%B %d %Y", "%B %d,%Y",
                            "%b %d, %Y", "%b %d %Y", "%b %d,%Y"):
                    try:
                        dt = datetime.strptime(date_str, fmt)
                        iso_dt = dt.strftime("%Y-%m-%d")
                        if time_str:
                            for tfmt in ("%I:%M %p", "%I:%M%p"):
                                try:
                                    t = datetime.strptime(
                                        time_str.upper().replace(".", ""), tfmt
                                    )
                                    iso_dt += t.strftime(" %H:%M")
                                    break
                                except ValueError:
                                    continue
                        break
                    except ValueError:
                        continue
                continue
            # First content line isn't a date → unparseable block
            return None

        # After date: tags or description
        clean = _DT_INLINE_NOISE.sub("", line).strip()
        if not clean:
            continue
        if _DT_TAG_LINE.match(clean):
            tags.append(clean.rstrip(","))
        else:
            desc_parts.append(clean)

    if not found_date:
        return None

    return {
        "title":       title,
        "date_str":    date_str,
        "time_str":    time_str,
        "iso_dt":      iso_dt,
        "description": " ".join(desc_parts),
        "tags":        tags,
    }


def _dt_parse_events_from_body(body: str) -> list[dict]:
    """
    Split body on H1 headings; parse each block into an event dict.
    parts = [pre-H1 noise, title1, body1, title2, body2, ...]
    """
    parts  = _DT_H1_SPLIT.split(body)
    events = []
    # parts[0] is pre-H1 content (page header) → skip
    for i in range(1, len(parts), 2):
        title      = parts[i].strip()
        block_text = parts[i + 1] if i + 1 < len(parts) else ""
        ev = _dt_parse_block(title, block_text)
        if ev:
            events.append(ev)
        else:
            print(f"  ⚠  Could not parse date for: {title!r}")
    return events


def _dt_build_chunk(ev: dict, month_n: int,
                    idx: int, global_idx: int, source_url: str) -> dict:
    month_label = _MONTH_LABEL.get(f"{month_n:02d}", str(month_n))
    date_line   = ev["date_str"]
    if ev["time_str"]:
        date_line += f" at {ev['time_str']}"

    text_lines = [f"Event Name: {ev['title']}", f"Date: {date_line}"]
    if ev["description"]:
        text_lines.append(f"Description: {ev['description']}")
    if ev["tags"]:
        text_lines.append(f"Categories: {', '.join(ev['tags'])}")
    text = "\n".join(text_lines)

    return {
        "chunk_id":    f"C_downtown_pgh_events_{month_n:02d}__{idx:04d}",
        "doc_id":      f"C_downtown_pgh_events_{month_n:02d}",
        "collection":  "C",
        "source":      "downtownpittsburgh.com",
        "url":         source_url,
        "title":       f"Downtown Pittsburgh Events — {month_label} 2026",
        "md_title":    f"Downtown Pittsburgh Events — {month_label} 2026",
        "section":     f"{ev['date_str'][:3] if ev['date_str'] else month_label} 2026",
        "subsection":  None,
        "chunk_index": global_idx,
        "doc_type":    "event",
        "event_date":  ev["iso_dt"],
        "venue":       "",      # downtown pages list venue inconsistently; omit
        "tags":        ev["tags"],
        "text":        text,
    }


def chunk_downtown_month(month_n: int, global_offset: int = 0) -> list[dict]:
    path = PROCESSED_DIR / f"downtown_pgh_events_{month_n:02d}.md"
    if not path.exists():
        print(f"  ⚠  Missing: {path.name} — skipping")
        return []
    source_url, body = _strip_yaml(path.read_text(encoding="utf-8"))
    events = _dt_parse_events_from_body(body)
    chunks = [
        _dt_build_chunk(ev, month_n, i, global_offset + i, source_url)
        for i, ev in enumerate(events)
    ]
    print(f"  ✓  {path.name:<45}  {len(events):>4} events  →  {len(chunks)} chunks")
    return chunks


# ──────────────────────────────────────────────
# Collection-level entry point
# ──────────────────────────────────────────────

def chunk_collection_c(
        pgh_months: list[str] = PGH_MONTHS,
        downtown_months: list[int] = DOWNTOWN_MONTHS,
        output_path: Path | None = None,
        append: bool = False,
) -> list[dict]:
    """
    Chunk all Pittsburgh Events sources and write C_chunks.jsonl.

    Args:
        pgh_months:      pittsburgh.events month slugs (default: all 10).
        downtown_months: downtownpittsburgh.com month numbers (default: 3–12).
        output_path:     Override default output path.
        append:          Append to existing file instead of overwriting.
                         Use append=True when merging with CMU event chunks.

    Returns:
        All chunk dicts from both sources.
    """
    output_path = output_path or PROCESSED_DIR / "C_chunks.jsonl"

    print("=" * 60)
    print("Chunking Collection C — Pittsburgh Events")
    print("=" * 60)

    all_chunks: list[dict] = []

    # ── pittsburgh.events ────────────────────────────────────────
    print("\n[pittsburgh.events]")
    for month in pgh_months:
        chunks = chunk_pgh_month(month, global_offset=len(all_chunks))
        all_chunks.extend(chunks)

    # ── downtownpittsburgh.com ────────────────────────────────────
    print("\n[downtownpittsburgh.com]")
    for n in downtown_months:
        chunks = chunk_downtown_month(n, global_offset=len(all_chunks))
        all_chunks.extend(chunks)

    # ── Write output ─────────────────────────────────────────────
    mode = "a" if append else "w"
    with open(output_path, mode, encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    size_kb = output_path.stat().st_size / 1024
    pgh_n      = sum(1 for c in all_chunks if c["source"] == "pittsburgh.events")
    downtown_n = sum(1 for c in all_chunks if c["source"] == "downtownpittsburgh.com")
    print(f"\n  → {'Appended' if append else 'Wrote'} {len(all_chunks)} total chunks "
          f"to {output_path}  ({size_kb:.1f} KB)")
    print(f"     pittsburgh.events      : {pgh_n}")
    print(f"     downtownpittsburgh.com : {downtown_n}")

    return all_chunks