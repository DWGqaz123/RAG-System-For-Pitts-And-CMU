"""
scrape_cmu.py
-------------
Scrape Carnegie Mellon University pages using Firecrawl API.
Raw output saved to scraped_data/raw/, cleaned Markdown to scraped_data/processed/.

Usage (import in notebook):
    from scrape_cmu import scrape_all, TARGETS
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
        "url": "https://www.cmu.edu/about/",
        "filename": "cmu_about.md",
    },
    {
        "url": "https://www.cmu.edu/about/vision-mission-values",
        "filename": "cmu_vision_mission_values.md",
    },
    {
        "url": "https://www.cmu.edu/leadership",
        "filename": "cmu_leadership.md",
    },
]

SCRAPE_OPTIONS = {
    "formats": ["markdown"],
    "onlyMainContent": True,
    "excludeTags": [
        "nav", "footer", "header", "aside",
        ".global-header",
        ".global-footer",
        ".breadcrumb",
        ".social-links",
        ".cookie-notice",
        ".skip-to-content",
        "#utility-nav",
    ],
    "waitFor": 1500,
}


# Section: Cleaning pipeline.

def clean_markdown(text: str) -> str:
    """
    Clean raw Firecrawl Markdown into plain text suitable for RAG chunking.

    Steps:
    1. Remove image tags               ![](...) / ![alt](...)
    2. Unwrap hyperlinks               [text](url) → text
    3. Remove bare URLs                http(s)://...
    4. Remove HTML comments            <!-- ... -->
    5. Remove inline HTML tags         <tag ...>
    6. Remove citation markers         [1], [note 2], [a], etc.
    7. Remove CMU-specific noise lines (nav labels, CTA text, etc.)
    8. Collapse excessive blank lines  (>2 → 1)
    9. Strip trailing whitespace
    """
    # Remove markdown images.
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)

    # Keep link text and drop URLs.
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

    # Remove CMU-specific boilerplate lines.
    noise_patterns = [
        r'^\s*Skip to (main )?content\s*$',
        r'^\s*Search\s*$',
        r'^\s*Menu\s*$',
        r'^\s*Toggle (navigation|menu)\s*$',
        r'^\s*Back to top\s*$',
        r'^\s*Share (this|page).*$',
        r'^\s*Follow (us|CMU).*$',
        r'^\s*Contact Us\s*$',
        r'^\s*©.*Carnegie Mellon.*$',
        r'^\s*All Rights Reserved.*$',
        r'^\s*Legal\s*$',
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
    """Call the Firecrawl /scrape endpoint and return the response JSON."""
    if not api_key:
        raise ValueError("api_key cannot be empty. Pass your Firecrawl API key.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {"url": url, **SCRAPE_OPTIONS}

    print(f"  → Requesting: {url}")
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
    """Write raw Markdown to scraped_data/raw/."""
    data     = result.get("data", {})
    markdown = data.get("markdown", "")
    metadata = data.get("metadata", {})
    path = RAW_DIR / target["filename"]
    path.write_text(_yaml_header(target, metadata) + markdown, encoding="utf-8")
    return path


def save_processed(result: dict, target: dict) -> Path:
    """Write cleaned Markdown to scraped_data/processed/."""
    data     = result.get("data", {})
    markdown = data.get("markdown", "")
    metadata = data.get("metadata", {})
    cleaned  = clean_markdown(markdown)
    path = PROCESSED_DIR / target["filename"]
    path.write_text(_yaml_header(target, metadata) + cleaned, encoding="utf-8")
    return path


def scrape_all(targets: list[dict], api_key: str, delay: float = 2.0) -> list[dict]:
    """
    Scrape all target URLs sequentially, saving both raw and cleaned versions.

    Args:
        targets:  List of dicts with 'url' and 'filename' keys.
        api_key:  Firecrawl API key (e.g. fc_api_key from notebook).
        delay:    Seconds to wait between requests to avoid rate limiting.

    Returns:
        List of result dicts with status, raw_path, and processed_path.
    """
    results = []
    for i, target in enumerate(targets):
        print(f"\n[{i+1}/{len(targets)}] Scraping: {target['url']}")
        try:
            raw_json       = scrape_url(target["url"], api_key)
            raw_path       = save_raw(raw_json, target)
            processed_path = save_processed(raw_json, target)

            raw_kb  = raw_path.stat().st_size / 1024
            proc_kb = processed_path.stat().st_size / 1024
            print(f"  ✓ Raw:       {raw_path}  ({raw_kb:.1f} KB)")
            print(f"  ✓ Processed: {processed_path}  ({proc_kb:.1f} KB)")

            results.append({
                "target":         target,
                "status":         "success",
                "raw_path":       str(raw_path),
                "processed_path": str(processed_path),
            })
        except requests.HTTPError as e:
            print(f"  ✗ HTTP error: {e.response.status_code} {e.response.text}")
            results.append({"target": target, "status": "error", "error": str(e)})
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results.append({"target": target, "status": "error", "error": str(e)})

        if i < len(targets) - 1:
            time.sleep(delay)

    return results
