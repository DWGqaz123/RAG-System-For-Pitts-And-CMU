"""
scrape_pdf_collection_b.py
--------------------------
Sources:
  1. Pittsburgh Payroll Tax Regulations
  2. Pittsburgh 2025 Operating Budget

Output:
    data/raw/B/payroll_tax_regulations.pdf
    data/raw/B/2025_operating_budget.pdf
    data/processed/B/payroll_tax_regulations.md
    data/processed/B/2025_operating_budget.md

Usage:
    from scrape_pdf_collection_b import process_all, TARGETS
    results = process_all(TARGETS)
"""

import re
import time
import requests
from datetime import datetime
from pathlib import Path

import pymupdf4llm   # install via: pip install pymupdf4llm


# Section: Configuration.

RAW_DIR       = Path("data/raw/B")
PROCESSED_DIR = Path("data/processed/B")
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = [
    {
        "url":      "https://www.pittsburghpa.gov/files/assets/city/v/1/finance/documents/"
                    "tax-forms/9626_payroll_tax_regulations.pdf",
        "filename": "payroll_tax_regulations",
        "title":    "City of Pittsburgh Payroll Tax Regulations",
        "source":   "pittsburghpa.gov",
    },
    {
        "url":      "https://www.pittsburghpa.gov/files/assets/city/v/4/omb/documents/"
                    "operating-budgets/2025-operating-budget.pdf",
        "filename": "2025_operating_budget",
        "title":    "City of Pittsburgh 2025 Operating Budget",
        "source":   "pittsburghpa.gov",
    },
]


# Section: Download.

def download_pdf(url: str, dest: Path) -> Path:
    """
    Download a PDF from url to dest.
    Skips download if the file already exists (idempotent).
    """
    if dest.exists():
        print(f"  ↩  Already downloaded: {dest.name}  ({dest.stat().st_size / 1024:.0f} KB)")
        return dest

    print(f"  ↓  Downloading: {url}")
    response = requests.get(url, timeout=120, stream=True)
    response.raise_for_status()

    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"  ✓  Saved: {dest.name}  ({dest.stat().st_size / 1024:.0f} KB)")
    return dest


# Section: Conversion.

def pdf_to_markdown(pdf_path: Path) -> str:
    """
    Convert a PDF to Markdown using pymupdf4llm.

    pymupdf4llm advantages for dense structured PDFs:
      - Preserves table structure as Markdown tables
      - Respects multi-column layouts
      - Extracts headings based on font size/weight
      - Handles page headers/footers via margins
    """
    md_text = pymupdf4llm.to_markdown(
        str(pdf_path),
        show_progress=True,
        # Ignore top/bottom margins to reduce repeated headers/footers.
        margins=(36, 50, 36, 50),   # (left, top, right, bottom) in points
    )
    return md_text


# Section: Cleaning pipeline.

def clean_markdown(text: str) -> str:
    """
    Light post-processing of pymupdf4llm output for RAG suitability.

    Cleaning steps:
    1. Remove pymupdf4llm page-break markers         -----
    2. Remove repeated page-number artefacts         e.g. "- 12 -", "Page 12"
    3. Strip footer/header lines (short repeated lines across pages)
    4. Normalise multiple blank lines                (>2 → 1)
    5. Strip trailing whitespace per line
    """
    # Remove page-break separator lines.
    text = re.sub(r'\n-{4,}\n', '\n\n', text)

    # Remove standalone page numbers.
    text = re.sub(r'^\s*[-–]?\s*\d{1,4}\s*[-–]?\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*Page\s+\d+\s+of\s+\d+\s*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r'^\s*Page\s+\d+\s*$', '', text, flags=re.MULTILINE | re.IGNORECASE)

    # Remove frequent footer/header artifacts.
    noise_patterns = [
        r'^\s*City of Pittsburgh\s*$',
        r'^\s*Office of Management and Budget\s*$',
        r'^\s*Department of Finance\s*$',
        r'^\s*pittsburghpa\.gov\s*$',
        r'^\s*CONFIDENTIAL\s*$',
        r'^\s*DRAFT\s*$',
    ]
    for pattern in noise_patterns:
        text = re.sub(pattern, '', text, flags=re.MULTILINE | re.IGNORECASE)

    # Collapse repeated blank lines.
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Trim trailing spaces per line.
    lines = [line.rstrip() for line in text.splitlines()]
    return '\n'.join(lines).strip()


# Section: YAML metadata.

def _yaml_header(target: dict) -> str:
    return (
        f"---\n"
        f"source_url: {target['url']}\n"
        f"scraped_at: {datetime.utcnow().isoformat()}Z\n"
        f"title: {target['title']}\n"
        f"source: {target['source']}\n"
        f"---\n\n"
    )


# Section: Main pipeline.

def process_pdf(target: dict) -> dict:
    """
    Full pipeline for a single PDF target:
      download → convert → clean → save raw md + processed md.
    """
    pdf_path = RAW_DIR / f"{target['filename']}.pdf"
    proc_path = PROCESSED_DIR / f"{target['filename']}.md"

    # Download source PDF.
    download_pdf(target["url"], pdf_path)

    # Convert PDF to markdown.
    print(f"  ⚙  Converting to Markdown: {pdf_path.name}")
    t0 = time.time()
    raw_md = pdf_to_markdown(pdf_path)
    print(f"  ✓  Converted in {time.time() - t0:.1f}s  ({len(raw_md):,} chars)")

    # Save raw markdown for inspection.
    raw_md_path = RAW_DIR / f"{target['filename']}.md"
    raw_md_path.write_text(_yaml_header(target) + raw_md, encoding="utf-8")

    # Apply text cleaning.
    cleaned = clean_markdown(raw_md)
    reduction = (1 - len(cleaned) / len(raw_md)) * 100
    print(f"  ✓  Cleaned: {len(raw_md):,} → {len(cleaned):,} chars  ({reduction:.0f}% reduction)")

    # Save cleaned markdown.
    proc_path.write_text(_yaml_header(target) + cleaned, encoding="utf-8")
    print(f"  ✓  Processed saved: {proc_path}  ({proc_path.stat().st_size / 1024:.0f} KB)")

    return {
        "target":         target,
        "status":         "success",
        "pdf_path":       str(pdf_path),
        "processed_path": str(proc_path),
        "raw_chars":      len(raw_md),
        "cleaned_chars":  len(cleaned),
    }


def process_all(targets: list[dict], delay: float = 1.0) -> list[dict]:
    """
    Process all PDF targets sequentially.

    Args:
        targets:  List of target dicts (url, filename, title, source).
        delay:    Seconds to wait between downloads.

    Returns:
        List of result dicts per target.
    """
    results = []
    for i, target in enumerate(targets):
        print(f"\n[{i+1}/{len(targets)}] {target['title']}")
        print(f"  URL: {target['url']}")
        try:
            result = process_pdf(target)
            results.append(result)
        except Exception as e:
            print(f"  ✗  Error: {e}")
            results.append({"target": target, "status": "error", "error": str(e)})

        if i < len(targets) - 1:
            time.sleep(delay)

    # Print run summary.
    print(f"\n{'='*60}")
    success = sum(1 for r in results if r["status"] == "success")
    print(f"Done: {success}/{len(results)} succeeded")
    for r in results:
        if r["status"] == "success":
            print(f"  ✓  {r['target']['filename']}.md  "
                  f"({r['cleaned_chars']:,} chars)")
        else:
            print(f"  ✗  {r['target']['filename']}  — {r.get('error')}")

    return results
