import csv
import time
import requests
from lxml import html
from pathlib import Path

INPUT_CSV  = "data/raw/D/visitpittsburgh_restaurants.csv"
OUTPUT_MD  = "data/processed/D/pittsburgh_restaurants.md"
DELAY      = 0.01  # seconds between requests

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def fetch_detail(url: str) -> str:
    """
    Fetch full description text from a restaurant detail page.
    Targets the div with class 'text--content lobotomize--level links',
    collecting all <p> tags within it and joining them into one paragraph.
    Falls back to meta description if the target div is not found.
    Returns an empty string on failure.
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        tree = html.fromstring(resp.content)

        # Primary path: collect paragraph text from the content block.
        paragraphs = tree.xpath(
            '//div[contains(@class,"text--content") and '
            'contains(@class,"lobotomize--level")]//p'
        )
        if paragraphs:
            texts = [p.text_content().strip() for p in paragraphs if p.text_content().strip()]
            if texts:
                return " ".join(texts)

        # Fallback path: use the meta description.
        meta = tree.xpath('//meta[@name="description"]/@content')
        if meta:
            return meta[0].strip()

        return ""

    except Exception as e:
        print(f"  [WARN] Failed to fetch {url}: {e}")
        return ""


def build_markdown(rows: list[dict]) -> str:
    """Render a list of restaurant dicts into a Markdown string."""
    lines = [
        "# Pittsburgh Restaurants Directory\n",
        f"*Source: visitpittsburgh.com — {len(rows)} restaurants*\n",
        "---\n",
    ]

    for row in rows:
        name    = row["name"]
        address = row["address"]
        phone   = row["phone"]
        link    = row["link"]
        summary = row["summary"]   # from CSV: map-infowindow__summary
        detail  = row["detail"]    # fetched from detail page (may be empty)

        lines += [
            f"## {name}\n",
            f"**Address:** {address}  ",
            f"**Phone:** {phone}  ",
            f"**Detail Page:** [{link}]({link})\n",
            f"**Summary:** {summary}\n",
            f"**Description:** {detail}\n" if detail else "**Description:** *(not available)*\n",
            "---\n",
        ]

    return "\n".join(lines)


def main():
    # Load source CSV rows.
    with open(INPUT_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        raw_rows = list(reader)

    print(f"Loaded {len(raw_rows)} restaurants. Scraping detail pages...\n")

    Path(OUTPUT_MD).parent.mkdir(parents=True, exist_ok=True)

    results = []
    for i, row in enumerate(raw_rows, 1):
        name    = row.get("map-infowindow__heading", "").strip()
        summary = row.get("map-infowindow__summary", "").strip()
        addr1   = row.get("card__address", "").strip()
        addr2   = row.get("card__address 2", "").strip()
        phone   = row.get("card__phone", "").strip()
        link    = row.get("detail_link", "").strip()

        address = ", ".join(p for p in [addr1, addr2] if p)

        print(f"[{i}/{len(raw_rows)}] {name}")
        detail = fetch_detail(link)

        if detail:
            print(f"  ✓ Scraped description ({len(detail)} chars)")
        else:
            print(f"  ✗ Could not retrieve description from detail page")

        results.append({
            "name": name, "address": address, "phone": phone,
            "link": link, "summary": summary, "detail": detail,
        })

        if i < len(raw_rows):
            time.sleep(DELAY)

    # Write combined markdown output.
    md_content = build_markdown(results)
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write(md_content)

    success = sum(1 for r in results if r["detail"])
    print(f"\nDone! {success}/{len(results)} descriptions scraped successfully.")
    print(f"Output saved to: {OUTPUT_MD}")


main()
