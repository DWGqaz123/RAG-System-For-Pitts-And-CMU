"""
scrape_visitpittsburgh.py
-------------------------
Scrape VisitPittsburgh pages with Firecrawl.
Save raw markdown to `scraped_data/raw/`.
Save cleaned markdown to `scraped_data/processed/`.

Usage:
    from scrape_visitpittsburgh import scrape_all, TARGETS
    results = scrape_all(TARGETS, api_key=fc_api_key)
"""

import re
import time
from datetime import datetime
from pathlib import Path
import requests


# Section: Configuration.

BASE_URL = "https://api.firecrawl.dev/v1"

RAW_DIR       = Path("scraped_data/raw")
PROCESSED_DIR = Path("scraped_data/processed")
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = [
    {
        "url": "https://www.visitpittsburgh.com/about-pittsburgh/",
        "filename": "visitpgh_about.md",
    },
    {
        "url": "https://www.visitpittsburgh.com/media/press-kit/pittsburgh-facts-trivia/",
        "filename": "visitpgh_facts_trivia.md",
    },
    {
        "url": "https://www.visitpittsburgh.com/about-pittsburgh/technology-in-pittsburgh/",
        "filename": "visitpgh_technology.md",
    },
    {
        "url": "https://www.visitpittsburgh.com/media/press-kit/pittsburgh-accolades/",
        "filename": "visitpgh_accolades.md",
    },
]

SCRAPE_OPTIONS = {
    "formats": ["markdown"],
    "onlyMainContent": True,
    "excludeTags": [
        "nav", "footer", "header", "aside",
        ".site-header", ".site-footer",
        ".newsletter-signup",
        ".social-share",
        ".cookie-banner",
        ".breadcrumb",
        ".pagination",
    ],
    "waitFor": 1500,
}


# Section: Cleaning pipeline.

def clean_markdown(text: str) -> str:
    """Clean markdown for indexing."""
    # Remove markdown images.
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)

    # Keep link text, drop URLs.
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)

    # Remove bare URLs.
    text = re.sub(r'https?://\S+', '', text)

    # Remove HTML comments.
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)

    # Remove inline HTML tags.
    text = re.sub(r'<[^>]+>', '', text)

    # Remove citation markers like [1], [note 2], [a].
    text = re.sub(r'\[\s*(?:[a-zA-Z\s]*)?\d+\s*\]', '', text)
    text = re.sub(r'\[\s*[a-zA-Z]\s*\]', '', text)

    # Remove VisitPittsburgh boilerplate lines.
    noise_patterns = [
        r'^\s*Learn More\s*$',
        r'^\s*Read More\s*$',
        r'^\s*View More\s*$',
        r'^\s*See More\s*$',
        r'^\s*Plan Your Visit.*$',
        r'^\s*Book Now\s*$',
        r'^\s*Subscribe.*$',
        r'^\s*Sign Up.*$',
        r'^\s*Follow Us.*$',
        r'^\s*Share This.*$',
        r'^\s*©.*VisitPittsburgh.*$',
        r'^\s*All Rights Reserved.*$',
        r'^\s*\|.*\|\s*$',          # markdown separator rows
    ]
    for pattern in noise_patterns:
        text = re.sub(pattern, '', text, flags=re.MULTILINE | re.IGNORECASE)

    # Collapse repeated blank lines.
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Trim trailing spaces per line.
    lines = [line.rstrip() for line in text.splitlines()]
    text = '\n'.join(lines).strip()

    return text


# Section: Scrape helpers.

def scrape_url(url: str, api_key: str) -> dict:
    """Call Firecrawl /scrape."""
    if not api_key:
        raise ValueError("api_key is required.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {"url": url, **SCRAPE_OPTIONS}

    print(f"  -> Requesting: {url}")
    response = requests.post(f"{BASE_URL}/scrape", headers=headers, json=payload, timeout=90)
    response.raise_for_status()
    return response.json()


def _yaml_header(target: dict, metadata: dict) -> str:
    return (
        f"---\n"
        f"source_url: {target['url']}\n"
        f"scraped_at: {datetime.utcnow().isoformat()}Z\n"
        f"title: {metadata.get('title', '')}\n"
        f"description: {metadata.get('description', '')}\n"
        f"---\n\n"
    )


def save_raw(result: dict, target: dict) -> Path:
    data     = result.get("data", {})
    markdown = data.get("markdown", "")
    metadata = data.get("metadata", {})
    path = RAW_DIR / target["filename"]
    path.write_text(_yaml_header(target, metadata) + markdown, encoding="utf-8")
    return path


def save_processed(result: dict, target: dict) -> Path:
    data     = result.get("data", {})
    markdown = data.get("markdown", "")
    metadata = data.get("metadata", {})
    cleaned  = clean_markdown(markdown)
    path = PROCESSED_DIR / target["filename"]
    path.write_text(_yaml_header(target, metadata) + cleaned, encoding="utf-8")
    return path


def scrape_all(targets: list[dict], api_key: str, delay: float = 2.0) -> list[dict]:
    """Scrape all targets and save raw/clean files."""
    results = []
    for i, target in enumerate(targets):
        print(f"\n[{i+1}/{len(targets)}] Scraping: {target['url']}")
        try:
            raw_json       = scrape_url(target["url"], api_key)
            raw_path       = save_raw(raw_json, target)
            processed_path = save_processed(raw_json, target)

            raw_kb  = raw_path.stat().st_size / 1024
            proc_kb = processed_path.stat().st_size / 1024
            print(f"  + Raw:   {raw_path}  ({raw_kb:.1f} KB)")
            print(f"  + Clean: {processed_path}  ({proc_kb:.1f} KB)")

            results.append({
                "target":         target,
                "status":         "success",
                "raw_path":       str(raw_path),
                "processed_path": str(processed_path),
            })
        except requests.HTTPError as e:
            print(f"  x HTTP error: {e.response.status_code} {e.response.text}")
            results.append({"target": target, "status": "error", "error": str(e)})
        except Exception as e:
            print(f"  x Error: {e}")
            results.append({"target": target, "status": "error", "error": str(e)})

        if i < len(targets) - 1:
            time.sleep(delay)

    return results
