"""
scrape_events_collection_c.py
------------------------------
Scrape Pittsburgh event pages for Collection C using Firecrawl API.

Four sources:
  1. pittsburgh.events          — monthly event listings (Mar–Dec)
  2. downtownpittsburgh.com     — downtown events by month (Mar–Dec 2026)
  3. events.cmu.edu             — CMU campus calendar by month (Mar–Dec 2026)
  4. cmu.edu/engage/events      — CMU alumni & community events

Pipeline per page:
  → Firecrawl /scrape → raw Markdown → clean → YAML front-matter
  → data/raw/<filename>.md
  → data/processed/<filename>.md

Usage (import in notebook):
    from scrape_events_collection_c import scrape_all, TARGETS
    results = scrape_all(TARGETS, api_key=fc_api_key)
"""

import re
import time
from datetime import datetime
from pathlib import Path
import requests


# Section: Configuration.

BASE_URL      = "https://api.firecrawl.dev/v1"
RAW_DIR       = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


_PGH_MONTHS = [
    "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
]

# Downtown month numbers (3=March ... 12=December).
_DOWNTOWN_MONTHS = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

_CMU_MONTHS = [
    "20260301", "20260401", "20260501", "20260601",
    "20260701", "20260801", "20260901", "20261001",
    "20261101", "20261201",
]

_MONTH_NAMES = {
    "03": "March",    "04": "April",    "05": "May",
    "06": "June",     "07": "July",     "08": "August",
    "09": "September","10": "October",  "11": "November",
    "12": "December",
}

TARGETS: list[dict] = []

# Generate one Pittsburgh Events target per month.
for month in _PGH_MONTHS:
    TARGETS.append({
        "url":      f"https://pittsburgh.events/{month}/",
        "filename": f"pgh_events_{month}",
        "title":    f"Pittsburgh Events — {month.capitalize()}",
        "source":   "pittsburgh.events",
        "group":    "pittsburgh_events",
    })

# Generate one Downtown Pittsburgh target per month in 2026.
for n in _DOWNTOWN_MONTHS:
    month_label = _MONTH_NAMES[f"{n:02d}"]
    TARGETS.append({
        "url":      f"https://downtownpittsburgh.com/events/?n={n}&y=2026&cat=0",
        "filename": f"downtown_pgh_events_{n:02d}",
        "title":    f"Downtown Pittsburgh Events — {month_label} 2026",
        "source":   "downtownpittsburgh.com",
        "group":    "downtown_pgh",
    })

# Generate one CMU campus-events target per month.
for yyyymmdd in _CMU_MONTHS:
    month_label = _MONTH_NAMES[yyyymmdd[4:6]]
    year        = yyyymmdd[:4]
    TARGETS.append({
        "url":      f"https://events.cmu.edu/month/date/{yyyymmdd}",
        "filename": f"cmu_campus_events_{yyyymmdd}",
        "title":    f"CMU Campus Events — {month_label} {year}",
        "source":   "events.cmu.edu",
        "group":    "cmu_campus",
    })

# Add the single CMU Engage events page.
TARGETS.append({
    "url":      "https://www.cmu.edu/engage/events",
    "filename": "cmu_engage_events",
    "title":    "CMU Engage Events — Alumni & Community",
    "source":   "cmu.edu",
    "group":    "cmu_engage",
})



_SCRAPE_OPTIONS: dict[str, dict] = {
    "pittsburgh_events": {
        "formats":         ["markdown"],
        "onlyMainContent": True,
        "excludeTags":     [
            "nav", "footer", "header", "aside",
            ".sidebar", ".widget", ".ad", ".newsletter",
            ".cookie-notice", ".social-links",
        ],
        "waitFor": 2000,
    },
    "downtown_pgh": {
        "formats":         ["markdown"],
        "onlyMainContent": True,
        "excludeTags":     [
            "nav", "footer", "header", "aside",
            ".site-header", ".site-footer",
            ".widget", ".sidebar",
            ".newsletter-signup", ".cookie-notice",
            ".wp-pagenavi",                    # pagination widget
            ".tribe-events-nav-pagination",    # plugin pagination
            ".tribe-events-calendar",          # calendar grid
            ".tribe-bar-filters",              # filter controls
        ],
        "waitFor": 2500,    # allow plugin-rendered content
    },
    "cmu_campus": {
        "formats":         ["markdown"],
        "onlyMainContent": True,
        "excludeTags":     [
            "nav", "footer", "header", "aside",
            "#utility-nav", ".global-header", ".global-footer",
            ".lw_cal_navigation",
            ".lw_widget_footer",
        ],
        "waitFor": 2500,
    },
    "cmu_engage": {
        "formats":         ["markdown"],
        "onlyMainContent": True,
        "excludeTags":     [
            "nav", "footer", "header", "aside",
            ".global-header", ".global-footer",
            ".breadcrumb", ".social-share",
        ],
        "waitFor": 1500,
    },
}


# Section: Cleaning pipeline.

def clean_markdown(text: str, group: str) -> str:
    """
    Clean raw Firecrawl Markdown into RAG-ready plain text.

    Common steps (all groups):
    1. Remove image tags
    2. Unwrap hyperlinks → keep display text
    3. Remove bare URLs
    4. Remove HTML comments and inline tags
    5. Remove citation markers
    6. Collapse excessive blank lines
    7. Strip trailing whitespace

    Group-specific noise patterns applied after common steps.
    """
    # Remove markdown images.
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    # Keep link text and drop URLs.
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    # Remove bare URLs.
    text = re.sub(r'https?://\S+', '', text)
    # Remove HTML comments and tags.
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    text = re.sub(r'<[^>]+>', '', text)
    # Remove citation markers like [1], [note 2], [a].
    text = re.sub(r'\[\s*(?:[a-zA-Z\s]*)?\d+\s*\]', '', text)
    text = re.sub(r'\[\s*[a-zA-Z]\s*\]', '', text)

    # Remove group-specific boilerplate lines.
    noise: list[str] = []
    if group == "pittsburgh_events":
        noise = [
            r'^\s*Filter\s*$',
            r'^\s*Sort By.*$',
            r'^\s*View (All|Calendar|List).*$',
            r'^\s*Load More.*$',
            r'^\s*Subscribe.*$',
            r'^\s*Add to Calendar\s*$',
            r'^\s*Share\s*$',
            r'^\s*\d+\s+events?\s*$',
        ]
    elif group == "downtown_pgh":
        noise = [
            r'^\s*Filter Events\s*$',
            r'^\s*All Categories\s*$',
            r'^\s*Find Events\s*$',
            r'^\s*Search\s*$',
            r'^\s*Previous Events?\s*$',
            r'^\s*Next Events?\s*$',
            r'^\s*Add to Calendar\s*$',
            r'^\s*\+ Google Calendar\s*$',
            r'^\s*\+ iCal Export\s*$',
            r'^\s*View Organizer Website\s*$',
            r'^\s*Venue Website\s*$',
            r'^\s*Get Directions\s*$',
            r'^\s*©.*Downtown Pittsburgh.*$',
            r'^\s*\d+\s+events?\s*$',
        ]
    elif group == "cmu_campus":
        noise = [
            r'^\s*Skip to (main )?content\s*$',
            r'^\s*Previous Month\s*$',
            r'^\s*Next Month\s*$',
            r'^\s*View Full Calendar\s*$',
            r'^\s*Add to (My )?Calendar\s*$',
            r'^\s*iCal.*$',
            r'^\s*Export.*$',
            r'^\s*©.*Carnegie Mellon.*$',
        ]
    elif group == "cmu_engage":
        noise = [
            r'^\s*Skip to (main )?content\s*$',
            r'^\s*Back to top\s*$',
            r'^\s*Share\s*$',
            r'^\s*Register\s*$',
            r'^\s*©.*Carnegie Mellon.*$',
        ]
    for pattern in noise:
        text = re.sub(pattern, '', text, flags=re.MULTILINE | re.IGNORECASE)

    # Normalize whitespace.
    text = re.sub(r'\n{3,}', '\n\n', text)
    lines = [line.rstrip() for line in text.splitlines()]
    return '\n'.join(lines).strip()


# Section: Scrape helpers.

def scrape_url(url: str, api_key: str, group: str) -> dict:
    """Call Firecrawl /scrape with group-specific options."""
    if not api_key:
        raise ValueError("api_key cannot be empty.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }
    payload = {"url": url, **_SCRAPE_OPTIONS[group]}

    print(f"  → {url}")
    response = requests.post(f"{BASE_URL}/scrape", headers=headers,
                             json=payload, timeout=90)
    response.raise_for_status()
    return response.json()


def _yaml_header(target: dict, metadata: dict) -> str:
    title = metadata.get("title") or target["title"]
    return (
        f"---\n"
        f"source_url: {target['url']}\n"
        f"scraped_at: {datetime.utcnow().isoformat()}Z\n"
        f"title: {title}\n"
        f"source: {target['source']}\n"
        f"group: {target['group']}\n"
        f"---\n\n"
    )


def save_raw(result: dict, target: dict) -> Path:
    """Save unmodified Markdown to data/raw/."""
    data     = result.get("data", {})
    markdown = data.get("markdown", "")
    metadata = data.get("metadata", {})
    path     = RAW_DIR / f"{target['filename']}.md"
    path.write_text(_yaml_header(target, metadata) + markdown, encoding="utf-8")
    return path


def save_processed(result: dict, target: dict) -> Path:
    """Save cleaned Markdown to data/processed/."""
    data     = result.get("data", {})
    markdown = data.get("markdown", "")
    metadata = data.get("metadata", {})
    cleaned  = clean_markdown(markdown, target["group"])
    path     = PROCESSED_DIR / f"{target['filename']}.md"
    path.write_text(_yaml_header(target, metadata) + cleaned, encoding="utf-8")
    return path


def scrape_one(target: dict, api_key: str) -> dict:
    """Scrape, clean, and save one target. Returns a result dict."""
    raw_json       = scrape_url(target["url"], api_key, target["group"])
    raw_path       = save_raw(raw_json, target)
    processed_path = save_processed(raw_json, target)

    raw_kb  = raw_path.stat().st_size / 1024
    proc_kb = processed_path.stat().st_size / 1024
    print(f"  ✓  raw: {raw_kb:.1f} KB  →  processed: {proc_kb:.1f} KB  [{target['filename']}]")

    return {
        "target":         target,
        "status":         "success",
        "raw_path":       str(raw_path),
        "processed_path": str(processed_path),
    }


# Section: Batch runner.

def scrape_all(targets: list[dict], api_key: str,
               delay: float = 2.0,
               skip_existing: bool = True) -> list[dict]:
    """
    Scrape all targets sequentially.

    Args:
        targets:        List of target dicts (use the module-level TARGETS).
        api_key:        Firecrawl API key.
        delay:          Seconds between requests.
        skip_existing:  Skip targets whose processed file already exists.

    Returns:
        List of result dicts (one per target).
    """
    results = []
    skipped = 0

    for i, target in enumerate(targets):
        proc_path = PROCESSED_DIR / f"{target['filename']}.md"

        if skip_existing and proc_path.exists():
            print(f"  ↩  [{i+1}/{len(targets)}] Skipping (exists): {target['filename']}")
            results.append({
                "target":         target,
                "status":         "skipped",
                "processed_path": str(proc_path),
            })
            skipped += 1
            continue

        print(f"\n[{i+1}/{len(targets)}] {target['title']}")
        try:
            result = scrape_one(target, api_key)
            results.append(result)
        except requests.HTTPError as e:
            print(f"  ✗  HTTP {e.response.status_code}: {e.response.text[:120]}")
            results.append({"target": target, "status": "error", "error": str(e)})
        except Exception as e:
            print(f"  ✗  Error: {e}")
            results.append({"target": target, "status": "error", "error": str(e)})

        if i < len(targets) - 1:
            time.sleep(delay)

    success = sum(1 for r in results if r["status"] == "success")
    errors  = sum(1 for r in results if r["status"] == "error")
    print(f"\n{'='*60}")
    print(f"Done:  {success} scraped  |  {skipped} skipped  |  {errors} errors")
    if errors:
        for r in results:
            if r["status"] == "error":
                print(f"  ✗  {r['target']['url']}  →  {r.get('error','')[:80]}")
    return results


# Section: Target filtering helper.

def targets_for_group(group: str) -> list[dict]:
    """Return only the targets belonging to a specific group."""
    return [t for t in TARGETS if t["group"] == group]
