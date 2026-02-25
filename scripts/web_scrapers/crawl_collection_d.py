"""
crawl_collection_d.py
======================
Collection D - Crawling script for local culture/food/sports websites in Pittsburgh

Use the FireCrawl v2 /crawl interface to initiate a crawl task for each main site.
Automatically discover sub-pages and save them as Markdown files.
-----------
data/raw/D/
  pghtacofest/
    pghtacofest_com_<hash>.md
    ...
  visitpittsburgh/
    visitpittsburgh_com_<hash>.md
    ...
  carnegiemuseums/
    ...
  ...

Each `.md` file has YAML front matter:
  ---
  source_url: https://...
  crawled_at: 2026-...
  doc_id: D_pghtacofest_0001
  ---

Usage
----
  python crawl_collection_d.py               # crawl all sites
  python crawl_collection_d.py --site taco   # crawl matching site_key only
  python crawl_collection_d.py --dry-run     # print tasks only
"""

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import requests


API_KEY    = "fc-8c63d293af3c42778cccfe4345935437"
BASE_URL   = "https://api.firecrawl.dev/v2"
HEADERS    = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
OUTPUT_DIR = Path("data/raw/D")

# Polling settings (seconds).
POLL_INTERVAL = 5
POLL_TIMEOUT  = 600   # 10 min


CRAWL_TASKS = [
    {
        "site_key":  "pghtacofest",
        "category":  "food",
        "url":       "https://www.pghtacofest.com/about",
        "limit":     20,
        "depth":     2,
        "include":   [],
        "exclude":   [],
    },
    {
        "site_key":  "visitpittsburgh",
        "category":  "food",           # also used for sports
        "url":       "https://www.visitpittsburgh.com/events-festivals/",
        "limit":     40,
        "depth":     3,
        "include":   [
            "/events-festivals/*",
            "/things-to-do/pittsburgh-sports-teams/*",
        ],
        "exclude":   [],
    },
    {
        "site_key":  "picklesburgh",
        "category":  "food",
        "url":       "https://www.picklesburgh.com/",
        "limit":     20,
        "depth":     2,
        "include":   [],
        "exclude":   [],
    },
    {
        "site_key":  "pittsburghrestaurantweek",
        "category":  "food",
        "url":       "https://pittsburghrestaurantweek.com/",
        "limit":     20,
        "depth":     2,
        "include":   [],
        "exclude":   [],
    },
    {
        "site_key":  "littleitalydays",
        "category":  "food",
        "url":       "https://littleitalydays.com/",
        "limit":     20,
        "depth":     2,
        "include":   [],
        "exclude":   [],
    },
    {
        "site_key":  "bananasplitfest",
        "category":  "food",
        "url":       "https://bananasplitfest.com/",
        "limit":     15,
        "depth":     2,
        "include":   [],
        "exclude":   [],
    },
    {
        "site_key":  "pittsburghsymphony",
        "category":  "music_culture",
        "url":       "https://www.pittsburghsymphony.org/pso_home/",
        "limit":     30,
        "depth":     3,
        "include":   [],
        "exclude":   [
            "/*news*",
            "/*blog*",
            "/*video*",
            "/*shop*",
        ],
    },
    {
        "site_key":  "pittsburghopera",
        "category":  "music_culture",
        "url":       "https://pittsburghopera.org/",
        "limit":     30,
        "depth":     3,
        "include":   [],
        "exclude":   [
            "/*news*",
            "/*blog*",
            "/*video*",
        ],
    },
    {
        "site_key":  "trustarts",
        "category":  "music_culture",
        "url":       "https://trustarts.org/",
        "limit":     30,
        "depth":     3,
        "include":   [],
        "exclude":   [
            "/*news*",
            "/*blog*",
        ],
    },
    {
        "site_key":  "carnegiemuseums",
        "category":  "music_culture",
        "url":       "https://carnegiemuseums.org/about-us/",
        "limit":     40,
        "depth":     3,
        "include":   [],
        "exclude":   [
            "/*news*",
            "/*video*",
            "/*store*",
            "/*shop*",
            "/*press*",
        ],
    },
    {
        "site_key":  "pirates",
        "category":  "sports",
        "url":       "https://www.mlb.com/pirates",
        "limit":     25,
        "depth":     2,
        "include":   ["/pirates/*"],
        "exclude":   [
            "/*scores*",
            "/*stats*",
            "/*standings*",
            "/*video*",
            "/*news*",
            "/*schedule*",   # too dynamic for static storage
        ],
    },
    {
        "site_key":  "steelers",
        "category":  "sports",
        "url":       "https://www.steelers.com/",
        "limit":     25,
        "depth":     2,
        "include":   [],
        "exclude":   [
            "/*scores*",
            "/*stats*",
            "/*standings*",
            "/*video*",
            "/*news*",
        ],
    },
    {
        "site_key":  "penguins",
        "category":  "sports",
        "url":       "https://www.nhl.com/penguins/",
        "limit":     25,
        "depth":     2,
        "include":   ["/penguins/*"],
        "exclude":   [
            "/*scores*",
            "/*stats*",
            "/*standings*",
            "/*video*",
            "/*news*",
        ],
    },
]

# Section: FireCrawl request helpers.

def _build_payload(task: dict) -> dict:
    """Build FireCrawl v2 /crawl payload."""
    exclude_paths = [
        "/*login*",
        "/*cart*",
        "/*checkout*",
        "/*signup*",
        "/*register*",
    ] + task.get("exclude", [])

    payload = {
        "url":                task["url"],
        "sitemap":            "skip",
        "crawlEntireDomain":  False,
        "limit":              task["limit"],
        "maxDiscoveryDepth":  task["depth"],
        "excludePaths":       exclude_paths,
        "scrapeOptions": {
            "onlyMainContent": True,
            "excludeTags": [
                "aside", "nav", "header", "footer",
                ".cookie-banner", ".cookie-notice",
                "#cookie", ".popup", ".modal",
            ],
            "maxAge":   172800000,   # 48h cache
            "formats":  ["markdown"],
            "parsers":  [],
        },
    }

    if task.get("include"):
        payload["includePaths"] = task["include"]

    return payload


def start_crawl(task: dict) -> str | None:
    """Start crawl and return job_id."""
    payload  = _build_payload(task)
    response = requests.post(f"{BASE_URL}/crawl", json=payload, headers=HEADERS, timeout=30)

    if response.status_code != 200:
        print(f"  [error] POST failed {response.status_code}: {response.text[:200]}")
        return None

    data   = response.json()
    job_id = data.get("id") or data.get("jobId")
    if not job_id:
        print(f"  [error] No job_id in response: {data}")
        return None

    print(f"  [started] job_id={job_id}")
    return job_id


def poll_crawl(job_id: str) -> list[dict] | None:
    """Poll crawl until completion."""
    deadline = time.time() + POLL_TIMEOUT
    while time.time() < deadline:
        time.sleep(POLL_INTERVAL)
        resp = requests.get(f"{BASE_URL}/crawl/{job_id}", headers=HEADERS, timeout=30)
        if resp.status_code != 200:
            print(f"  [error] poll {resp.status_code}: {resp.text[:200]}")
            return None

        data   = resp.json()
        status = data.get("status", "")
        total  = data.get("total", "?")
        done   = data.get("completed", "?")
        print(f"  [poll] status={status}  {done}/{total} pages")

        if status == "completed":
            return data.get("data", [])
        if status in ("failed", "cancelled"):
            print(f"  [error] job ended with status={status}")
            return None

    print(f"  [error] timeout after {POLL_TIMEOUT}s")
    return None



def _slug(url: str) -> str:
    """URL to safe filename."""
    url = re.sub(r"^https?://(?:www\.)?", "", url)
    url = re.sub(r"[^\w\-/.]", "_", url)
    url = url.strip("/").replace("/", "__")
    return url[:80] or "index"


def save_pages(pages: list[dict], task: dict, run_ts: str) -> int:
    """Save pages as markdown files."""
    out_dir = OUTPUT_DIR / task["site_key"]
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for i, page in enumerate(pages):
        url      = page.get("url") or page.get("metadata", {}).get("url", "")
        markdown = page.get("markdown") or page.get("content", "")

        if not markdown or not markdown.strip():
            continue

        slug     = _slug(url)
        filename = out_dir / f"{slug}.md"

        # Build YAML front matter for each saved page.
        doc_id   = f"D_{task['site_key']}_{i:04d}"
        header   = (
            "---\n"
            f"source_url: {url}\n"
            f"site_key: {task['site_key']}\n"
            f"category: {task['category']}\n"
            f"doc_id: {doc_id}\n"
            f"crawled_at: {run_ts}\n"
            "---\n\n"
        )
        filename.write_text(header + markdown, encoding="utf-8")
        saved += 1

    return saved



def crawl_all(tasks: list[dict], dry_run: bool = False) -> dict[str, int]:
    """Run all crawl tasks serially."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_ts  = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    results = {}

    print(f"{'='*60}")
    print(f"Collection D Crawl  —  {len(tasks)} sites")
    print(f"{'='*60}")

    for task in tasks:
        key = task["site_key"]
        print(f"\n[{key}]  {task['url']}")

        if dry_run:
            payload = _build_payload(task)
            print(f"  DRY RUN — payload preview:")
            print(f"    limit={payload['limit']}  depth={payload['maxDiscoveryDepth']}")
            print(f"    excludePaths={payload['excludePaths']}")
            if "includePaths" in payload:
                print(f"    includePaths={payload['includePaths']}")
            results[key] = 0
            continue

        # Respect free-plan rate limit (2 requests/minute).
        if results:   # skip waiting before the first task
            print("  [rate limit] waiting 62s before next request...")
            time.sleep(62)

        job_id = start_crawl(task)
        if not job_id:
            results[key] = -1
            continue

        pages = poll_crawl(job_id)
        if pages is None:
            results[key] = -1
            continue

        saved = save_pages(pages, task, run_ts)
        print(f"  [done] {len(pages)} pages fetched, {saved} saved  →  data/raw/D/{key}/")
        results[key] = saved

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    total_saved = 0
    for key, cnt in results.items():
        status = f"{cnt} files" if cnt >= 0 else "FAILED"
        print(f"  {key:<30} {status}")
        if cnt > 0:
            total_saved += cnt
    print(f"\n  Total saved: {total_saved} markdown files")
    print(f"  Output dir:  {OUTPUT_DIR.resolve()}")

    return results

def main():
    parser = argparse.ArgumentParser(description="Crawl Collection D websites via FireCrawl")
    parser.add_argument("--site",    type=str, default="",    help="crawl site_keys containing this string")
    parser.add_argument("--dry-run", action="store_true",     help="print payloads only")
    args = parser.parse_args()

    tasks = CRAWL_TASKS
    if args.site:
        tasks = [t for t in tasks if args.site.lower() in t["site_key"].lower()]
        print(f"Filtered to {len(tasks)} tasks matching '{args.site}'")

    crawl_all(tasks, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
