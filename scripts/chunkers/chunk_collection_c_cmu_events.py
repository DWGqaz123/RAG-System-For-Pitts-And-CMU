r"""
chunk_collection_c_cmu_events.py
=================================
Collection C — CMU campus events chunking script

Supports two file types
-----------------------
1. cmu_campus_events_YYYYMMDD.md (events.cmu.edu monthly calendar)

    Firecrawl renders the Localist calendar as a seven-column Markdown table.
    Each pipe-separated cell contains all events for a day, concatenated on a
    single line with no delimiters, for example:

         1 WedNoon \- 1:30 p.m.Libraries Workshop: Intro to R5 \- 6:30 p.m.Study Abroad...

    Parsing strategy:
      a. Find the "##### Month YYYY" start line and end at "### Previous Month".
      b. Split each line on "|" to get cells.
      c. For each cell:
          - Remove the date header (e.g., "29 Sun" so the time token is first).
          - Clean Markdown links, including broken links with escaped brackets.
          - Remove Weekly/Yearly/Daily/Monthly recurrence labels.
          - Mask date-span banners (e.g., "Apr 13 - 17Fall 2026 Registration";
             these multi-day notices have no clock time and are masked out).
          - Split on clock-time tokens to get (time, title) pairs.
      d. Each (day, time, title) triple becomes one chunk.

    Clock tokens are restricted to hours 1-12 to avoid mis-parsing strings like
    "127:30" inside "10 Out of 12" style text.

2. cmu_engage_events.md (cmu.edu/engage/events alumni/community page)

    Simple structure: any '#' heading opens a new section; the body becomes one
    chunk.

Output
------
    data/processed/C/C_docs.jsonl    (append)
    data/processed/C/C_chunks.jsonl  (append)

Usage (Notebook)
----------------
    from chunk_collection_c_cmu_events import run
    run()
    # Only process specific months
    run(campus_months=["20260301", "20260401"])
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Path config
# ---------------------------------------------------------------------------

PROCESSED_DIR = Path("data/processed/C")
OUT_DOCS      = PROCESSED_DIR / "C_docs.jsonl"
OUT_CHUNKS    = PROCESSED_DIR / "C_chunks.jsonl"

DEFAULT_CAMPUS_MONTHS = [
    "20260301", "20260401", "20260501", "20260601",
    "20260701", "20260801", "20260901", "20261001",
    "20261101", "20261201",
]


# ---------------------------------------------------------------------------
# Common utilities
# ---------------------------------------------------------------------------

_YAML_RE = re.compile(r"^---\n.*?\n---\n", re.DOTALL)

_MONTH_TO_NUM = {
    "january":   "01", "february": "02", "march":    "03", "april":    "04",
    "may":       "05", "june":     "06", "july":     "07", "august":   "08",
    "september": "09", "october":  "10", "november": "11", "december": "12",
}
_NUM_TO_MONTH = {v: k for k, v in _MONTH_TO_NUM.items()}


def _strip_yaml(text: str) -> tuple[str, str]:
    """Return (source_url, body_without_frontmatter)."""
    m    = re.search(r"^source_url:\s*(.+)$", text, re.MULTILINE)
    url  = m.group(1).strip() if m else ""
    body = _YAML_RE.sub("", text).strip()
    return url, body


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Campus calendar — regex constants
# ---------------------------------------------------------------------------

# 12-hour clock valid hours 1-12 (avoid matching "127:30")
_H = r"(?:1[0-2]|0?[1-9])"

# Clock tokens; longest matches first (ranges before single times)
_CLOCK_RE = re.compile(
    r"(?:"
    # Noon \- H pm          "Noon \- 1:30 p.m."
    rf"Noon\s*(?:\\-|[-–])\s*{_H}(?::\d{{2}})?\s*[ap]\.?m\.?"
    # H am \- Noon          "11 a.m. \- Noon"
    rf"|{_H}(?::\d{{2}})?\s*[ap]\.?m\.?\s*(?:\\-|[-–])\s*Noon"
    # H am \- H am/pm       "11 a.m. \- 12:30 p.m."
    rf"|{_H}(?::\d{{2}})?\s*[ap]\.?m\.?\s*(?:\\-|[-–])\s*{_H}(?::\d{{2}})?\s*[ap]\.?m\.?"
    # H \- H am/pm          "9 \- 11 a.m."
    rf"|{_H}(?::\d{{2}})?\s*(?:\\-|[-–])\s*{_H}(?::\d{{2}})?\s*[ap]\.?m\.?"
    # Single time           "5:30 p.m."
    rf"|{_H}(?::\d{{2}})?\s*[ap]\.?m\.?"
    r"|Noon|Midnight|All\s+Day"
    r")",
    re.IGNORECASE,
)

# Date-span tokens (multi-day banners without clock times)
# Examples: Apr 13 - 17 / Feb 13, 1 a.m. - Apr 5, Midnight
_DATE_SPAN_RE = re.compile(
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*"
    r"\s+\d{1,2}(?:,\s*\d{4})?(?:,\s*\d{1,2}(?::\d{2})?\s*[ap]\.?m\.?)?"
    r"\s*[-–]\s*"
    r"(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+)?\d{1,2}"
    r"(?:,\s*\d{4})?(?:,\s*\w+)?",
    re.IGNORECASE,
)

_RECUR_RE   = re.compile(r"(?:Weekly|Yearly|Daily|Monthly)\s*(?:\([^)]*\))?", re.IGNORECASE)
_ANDMORE_RE = re.compile(r"\s*and\s+\d+\s+more\.?\.?\.?", re.IGNORECASE)

# Date header: 29 Sun / 1 Wed — directly followed by times, no space
_DAY_HDR_RE = re.compile(
    r"^\d{1,2}\s*(?:Sun|Mon|Tue|Wed|Thu|Fri|Sat)(?=\d|\W|$)",
    re.IGNORECASE,
)

# Markdown link cleanup
_STRAY_LINK = re.compile(r"\[([^\]]*)\]")          # Remove leftover [word]
_LINK_OPEN  = re.compile(r"\[(?P<t>(?:[^\[\]\\]|\\.)*)\]\(", re.DOTALL)

# Broken-link URL termination boundary (protect following event titles)
_URL_END_RE = re.compile(
    rf"(?:and\s+\d+\s+more|{_H}(?::\d{{2}})?\s*[ap]\.?m|Noon[\s\\]|Midnight|All\s+Day)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Cell parsing helpers
# ---------------------------------------------------------------------------

def _replace_links(text: str) -> str:
    """
    Replace [display](url) with the display text, handling three cases:
      1. Normal links     [text](https://...)
      2. Escaped brackets [text \[Virtual\]](url)
      3. Broken links     [text]( \- 12:45 p.m.NextTitle...
         The URL is consumed only up to the next clock token to keep later titles.
    """
    result, pos = [], 0
    for m in _LINK_OPEN.finditer(text):
        result.append(text[pos:m.start()])
        display   = re.sub(r"\\(.)", r"\1", m.group("t"))
        url_start = m.end()
        rest      = text[url_start:]
        close     = rest.find(")")
        boundary  = _URL_END_RE.search(rest)
        if close != -1 and (boundary is None or close <= boundary.start()):
            url_end = url_start + close + 1         # Normal link
        elif boundary is not None:
            url_end = url_start + boundary.start()  # Broken link
        else:
            url_end = len(text)
        result.append(display)
        pos = url_end
    result.append(text[pos:])
    return "".join(result)


def _mask_spans(text: str) -> str:
    """
    Replace each date-span token plus its following title with spaces.
    Prevents multi-day banners (without clock times) from polluting titles.
    """
    result, pos = [], 0
    for m in _DATE_SPAN_RE.finditer(text):
        result.append(text[pos:m.start()])
        after = text[m.end():]
        nxt   = _CLOCK_RE.search(after)
        pos   = m.end() + (nxt.start() if nxt else len(after))
        result.append(" ")
    result.append(text[pos:])
    return "".join(result)


def _clean_title(t: str) -> str:
    """Clean leftover Firecrawl artifacts from titles."""
    t = re.sub(r"\(\s*(?:\\-|[-–])?\s*$", "", t)  # Trailing broken link fragment
    t = _RECUR_RE.sub("", t)                        # Residual recurrence labels
    t = t.strip(" .,;")
    # Drop a leading '(' only when parentheses are unbalanced; keep cases like "(Women's Tennis)"
    if t.startswith("(") and t.count("(") > t.count(")"):
        t = t[1:].strip()
    return t


def _parse_cell(raw: str) -> list[tuple[str, str]]:
    """Extract a list of (time_str, title) pairs from a concatenated cell."""
    body = _DAY_HDR_RE.sub("", raw.strip())  # Remove date header "29 Sun"
    body = _replace_links(body)              # Clean Markdown links
    body = _STRAY_LINK.sub(r"\1", body)      # Remove leftover [word]
    body = _RECUR_RE.sub("", body)           # Remove recurrence labels
    body = _ANDMORE_RE.sub("", body)         # Drop "and N more..."
    body = _mask_spans(body)                 # Mask multi-day banners
    toks = list(_CLOCK_RE.finditer(body))
    out  = []
    for i, tok in enumerate(toks):
        title = body[tok.end(): toks[i + 1].start() if i + 1 < len(toks) else len(body)]
        title = _clean_title(title)
        if title and len(title) >= 3:
            out.append((tok.group(), title))
    return out


# ---------------------------------------------------------------------------
# Campus calendar — per-month processing
# ---------------------------------------------------------------------------

def _parse_campus_calendar(body: str, yyyymmdd: str) -> list[dict]:
    """
    Extract all events from the monthly calendar Markdown body.
    Start at: ##### Month YYYY
    Stop at:  ### Previous Month
    """
    year       = yyyymmdd[:4]
    month_num  = yyyymmdd[4:6]
    month_name = _NUM_TO_MONTH.get(month_num, "").capitalize()

    start_pat = re.compile(rf"#{{2,5}}\s+{month_name}\s+{year}", re.IGNORECASE)
    end_pat   = re.compile(r"###\s+Previous Month", re.IGNORECASE)

    lines     = body.splitlines()
    start_idx = None
    end_idx   = len(lines)

    for i, ln in enumerate(lines):
        if start_idx is None and start_pat.search(ln):
            start_idx = i
        elif start_idx is not None and end_pat.search(ln):
            end_idx = i
            break

    if start_idx is None:
        print(f"  warning: heading '{month_name} {year}' not found; skipping")
        return []

    table_sep = re.compile(r"^[\s|:\-]+$")
    events: list[dict] = []

    for raw_line in lines[start_idx:end_idx]:
        s = raw_line.strip()
        if not s or s.startswith("#") or table_sep.match(s) or "|" not in s:
            continue
        for cell in (c.strip() for c in s.split("|") if c.strip()):
            hdr = re.match(
                r"^(\d{1,2})\s*(Sun|Mon|Tue|Wed|Thu|Fri|Sat)",
                cell, re.IGNORECASE,
            )
            if not hdr:
                continue
            day_num = hdr.group(1)
            weekday = hdr.group(2).capitalize()
            for time_str, title in _parse_cell(cell):
                events.append({
                    "day":        day_num,
                    "weekday":    weekday,
                    "month":      month_name,
                    "year":       year,
                    "date_label": f"{month_name} {day_num}, {year}",
                    "time":       time_str,
                    "title":      title,
                })
    return events


def process_campus_month(yyyymmdd: str) -> tuple[list[dict], list[dict]]:
    """Process one calendar file; return (docs, chunks)."""
    path = PROCESSED_DIR / f"cmu_campus_events_{yyyymmdd}.md"
    if not path.exists():
        print(f"  warning: missing file {path.name}; skipping")
        return [], []

    url, body  = _strip_yaml(path.read_text(encoding="utf-8"))
    events     = _parse_campus_calendar(body, yyyymmdd)
    month_num  = yyyymmdd[4:6]
    month_name = _NUM_TO_MONTH.get(month_num, "").capitalize()
    year       = yyyymmdd[:4]
    title_str  = f"CMU Campus Events — {month_name} {year}"

    doc = {
        "doc_id":       f"C_cmu_campus_events_{yyyymmdd}",
        "collection":   "C",
        "source":       "events.cmu.edu",
        "url":          url,
        "title":        title_str,
        "doc_type":     "event_calendar",
        "month":        f"{month_name} {year}",
        "event_count":  len(events),
        "scraped_at":   _now_iso(),
    }

    chunks = []
    for i, ev in enumerate(events):
        text = (
            f"Event: {ev['title']}\n"
            f"Date: {ev['weekday']}, {ev['date_label']}\n"
            f"Time: {ev['time']}\n"
            f"Source: CMU Campus Events"
        )
        chunks.append({
            "chunk_id":    f"C_cmu_campus_{yyyymmdd}__{i:04d}",
            "doc_id":      f"C_cmu_campus_events_{yyyymmdd}",
            "collection":  "C",
            "source":      "events.cmu.edu",
            "url":         url,
            "title":       title_str,
            "section":     f"{month_name} {year}",
            "subsection":  ev["date_label"],
            "chunk_index": i,
            "doc_type":    "event",
            "event_date":  ev["date_label"],
            "venue":       "CMU Campus",
            "text":        text,
        })

    print(f"  ok  {path.name:<45}  {len(events):>4} events  ->  {len(chunks)} chunks")
    return [doc], chunks


# ---------------------------------------------------------------------------
# Engage page — heading-based split
# ---------------------------------------------------------------------------

_HDR_RE   = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)
_SKIP_HDR = re.compile(
    r"^(previous|next|jump to|mini calendar|month view)",
    re.IGNORECASE,
)


def process_engage() -> tuple[list[dict], list[dict]]:
    """Process cmu_engage_events.md; return (docs, chunks)."""
    path = PROCESSED_DIR / "cmu_engage_events.md"
    if not path.exists():
        print(f"  warning: missing file {path.name}; skipping")
        return [], []

    url, body = _strip_yaml(path.read_text(encoding="utf-8"))
    parts     = _HDR_RE.split(body)
    sections  = []
    for i in range(1, len(parts), 3):
        title   = parts[i + 1].strip()
        content = parts[i + 2].strip() if i + 2 < len(parts) else ""
        if not title or len(title) < 3 or _SKIP_HDR.match(title):
            continue
        content = re.sub(r"!\[.*?\]\(.*?\)", "", content)
        content = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", content)
        content = re.sub(r"https?://\S+", "", content)
        content = re.sub(r"^\|.*$", "", content, flags=re.MULTILINE)
        content = re.sub(r"^[-| :]+$", "", content, flags=re.MULTILINE)
        content = re.sub(r"\n{3,}", "\n\n", content).strip()
        sections.append({"title": title, "content": content})

    doc = {
        "doc_id":        "C_cmu_engage_events",
        "collection":    "C",
        "source":        "cmu.edu",
        "url":           url,
        "title":         "CMU Engage Events — Alumni & Community",
        "doc_type":      "event_listing",
        "section_count": len(sections),
        "scraped_at":    _now_iso(),
    }

    chunks = []
    for i, sec in enumerate(sections):
        text = f"Event: {sec['title']}"
        if sec["content"]:
            text += f"\nDetails: {sec['content']}"
        chunks.append({
            "chunk_id":    f"C_cmu_engage__{i:04d}",
            "doc_id":      "C_cmu_engage_events",
            "collection":  "C",
            "source":      "cmu.edu",
            "url":         url,
            "title":       "CMU Engage Events — Alumni & Community",
            "section":     sec["title"],
            "subsection":  None,
            "chunk_index": i,
            "doc_type":    "event",
            "event_date":  None,
            "venue":       "",
            "text":        text,
        })

    print(f"  ok  {path.name:<45}  {len(sections):>4} sections  ->  {len(chunks)} chunks")
    return [doc], chunks


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(
    campus_months: list[str] = DEFAULT_CAMPUS_MONTHS,
    append: bool = True,
) -> tuple[list[dict], list[dict]]:
    """
    Process all CMU event files and write C_docs.jsonl and C_chunks.jsonl.

    Parameters
    ----------
    campus_months : List of YYYYMMDD strings for cmu_campus_events_YYYYMMDD.md
    append        : True = append (merge with other C sources), False = overwrite

    Returns
    -------
    (all_docs, all_chunks)
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Collection C — CMU Events Chunking")
    print("=" * 60)

    all_docs:   list[dict] = []
    all_chunks: list[dict] = []

    print("\n[events.cmu.edu — campus calendar]")
    for yyyymmdd in campus_months:
        docs, chunks = process_campus_month(yyyymmdd)
        all_docs.extend(docs)
        all_chunks.extend(chunks)

    print("\n[cmu.edu/engage/events]")
    docs, chunks = process_engage()
    all_docs.extend(docs)
    all_chunks.extend(chunks)

    mode = "a" if append else "w"
    with open(OUT_DOCS, mode, encoding="utf-8") as f:
        for d in all_docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    with open(OUT_CHUNKS, mode, encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    docs_kb   = OUT_DOCS.stat().st_size   / 1024
    chunks_kb = OUT_CHUNKS.stat().st_size / 1024
    campus_n  = sum(1 for c in all_chunks if c["source"] == "events.cmu.edu")
    engage_n  = len(all_chunks) - campus_n

    print(f"\n  -> {'append' if append else 'write'} done")
    print(f"     C_docs.jsonl   : {len(all_docs)} docs  ({docs_kb:.1f} KB)")
    print(f"     C_chunks.jsonl : {len(all_chunks)} chunks  ({chunks_kb:.1f} KB)")
    print(f"       events.cmu.edu campus : {campus_n}")
    print(f"       cmu.edu engage        : {engage_n}")

    return all_docs, all_chunks


if __name__ == "__main__":
    run()