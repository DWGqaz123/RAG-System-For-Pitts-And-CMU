"""
chunk_collection_d.py
======================
Collection D — Local culture/food/sports site Markdown chunking script

Input
-----
    data/processed/D/<site_key>/<slug>.md   (cleaned output from clean_collection_d.py)

Chunking strategy
-----------------
1. Split on H1 / H2 / H3 headings (H4+ is too granular and merges into parent).
2. Each chunk carries:
         - Breadcrumb path: parent title > current title (helps embeddings understand context)
         - Body content
         - Overlap: up to OVERLAP_WORDS leading words from the next section to improve cross-section recall
3. Skip sections whose body word count is below MIN_WORDS.
4. A whole-document body with no headings is still emitted as one chunk.

Output format (text field)
--------------------------
    [Site: Little Italy Days | Category: food]
    Section: About the Festival > Vendors > Food Vendors

    We welcome food vendors offering Italian cuisine...
    (overlap) Handmade crafts, artwork...

Output files
------------
    data/processed/D_docs.jsonl    (one record per .md file)
    data/processed/D_chunks.jsonl  (one record per section/chunk)

Usage
-----
    python chunk_collection_d.py                  # process all files
    python chunk_collection_d.py --site carnegie  # only process a specific site
    python chunk_collection_d.py --dry-run        # print stats without writing files
"""

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths & parameters
# ---------------------------------------------------------------------------

PROCESSED_D_DIR = Path("data/processed/D")
OUT_DOCS        = Path("data/processed/D_docs.jsonl")
OUT_CHUNKS      = Path("data/processed/D_chunks.jsonl")

# Chunking parameters
SPLIT_LEVELS   = {1, 2, 3}   # Split on H1/H2/H3
MIN_WORDS      = 15           # Skip sections with fewer words
OVERLAP_WORDS  = 80           # Overlap words from the start of the next section


# ---------------------------------------------------------------------------
# YAML frontmatter parsing
# ---------------------------------------------------------------------------

_YAML_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Return (meta_dict, body); meta is empty dict when no frontmatter."""
    m = _YAML_RE.match(text)
    if not m:
        return {}, text
    meta = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            meta[k.strip()] = v.strip()
    body = text[m.end():]
    return meta, body


# ---------------------------------------------------------------------------
# Markdown heading parsing & section splitting
# ---------------------------------------------------------------------------

_HDR_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


def _heading_level(hashes: str) -> int:
    return len(hashes)


def _strip_md_syntax(text: str) -> str:
    """Remove leftover Markdown syntax to produce plain text."""
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)   # bold/italic
    text = re.sub(r"`([^`]+)`", r"\1", text)                # inline code
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)  # Remove heading markers
    text = re.sub(r"^\s*[-*+]\s+", "- ", text, flags=re.MULTILINE)  # Normalize bullet lists
    text = re.sub(r"^\s*\d+\.\s+", "- ", text, flags=re.MULTILINE)  # Normalize ordered lists
    text = re.sub(r"\|[-: |]+\|", "", text)                 # Drop table separator rows
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _word_count(text: str) -> int:
    return len(text.split())


def _overlap_text(text: str, max_words: int) -> str:
    """Take the first max_words words, trying to cut after sentence boundaries."""
    if not text.strip():
        return ""
    words = text.split()
    if len(words) <= max_words:
        return text.strip()
    snippet = " ".join(words[:max_words])
    # Prefer truncating at the last sentence-ending punctuation
    cut = max(snippet.rfind(". "), snippet.rfind("! "), snippet.rfind("? "))
    if cut > max_words // 2:
        snippet = snippet[:cut + 1]
    return snippet.strip()


class Section:
    """Represents a heading section."""
    __slots__ = ("level", "title", "body", "ancestors")

    def __init__(self, level: int, title: str, body: str, ancestors: list[str]):
        self.level     = level
        self.title     = title
        self.body      = body           # Raw markdown body
        self.ancestors = ancestors      # Parent headings (excluding self)


def _split_sections(body: str) -> list[Section]:
    """
    Split the body at heading levels in SPLIT_LEVELS and return Section objects.
    H4+ content is merged into the nearest H3 (or higher) ancestor.
    """
    # Find all heading positions
    all_hdrs = [(m.start(), _heading_level(m.group(1)), m.group(2).strip())
                for m in _HDR_RE.finditer(body)]

    # Keep only split points at the configured heading levels
    split_hdrs = [(pos, lvl, title)
                  for pos, lvl, title in all_hdrs
                  if lvl in SPLIT_LEVELS]

    sections: list[Section] = []

    # Preamble: content before the first heading — always included as its own
    # chunk regardless of word count, since it often contains the page intro or
    # a short description that is valuable for retrieval.
    first_pos = split_hdrs[0][0] if split_hdrs else len(body)
    preamble  = body[:first_pos].strip()
    if preamble:
        sections.append(Section(0, "(intro)", preamble, []))

    # Extract sections at split points
    ancestor_stack: list[tuple[int, str]] = []  # [(level, title), ...]

    for i, (pos, lvl, title) in enumerate(split_hdrs):
        end   = split_hdrs[i + 1][0] if i + 1 < len(split_hdrs) else len(body)
        chunk_body = body[pos:end]

        # Remove the heading line itself
        first_nl = chunk_body.find("\n")
        chunk_body = chunk_body[first_nl + 1:].strip() if first_nl != -1 else ""

        # Maintain ancestor stack
        ancestor_stack = [(l, t) for l, t in ancestor_stack if l < lvl]
        ancestors = [t for _, t in ancestor_stack]
        ancestor_stack.append((lvl, title))

        if _word_count(chunk_body) >= MIN_WORDS:
            sections.append(Section(lvl, title, chunk_body, ancestors))

    return sections


# ---------------------------------------------------------------------------
# Chunk construction
# ---------------------------------------------------------------------------

_SITE_TITLE_MAP: dict[str, str] = {}   # site_key → friendly title (inferred from H1)


def _infer_site_title(body: str, site_key: str) -> str:
    """Infer a friendly site name from the document H1 or the site_key."""
    m = re.search(r"^#\s+(.+)$", body, re.MULTILINE)
    if m:
        return m.group(1).strip()
    return site_key.replace("_", " ").replace("-", " ").title()


def _build_chunk_text(
    site_title: str,
    category: str,
    section: Section,
    overlap: str,
) -> str:
    """Assemble the chunk text field (plain text for embedding)."""
    # Breadcrumb
    breadcrumb_parts = section.ancestors + ([section.title] if section.title else [])
    breadcrumb = " > ".join(breadcrumb_parts) if breadcrumb_parts else "(intro)"

    header = f"[Site: {site_title} | Category: {category}]\nSection: {breadcrumb}"
    body   = _strip_md_syntax(section.body)

    text = header + "\n\n" + body
    if overlap:
        text += "\n\n" + overlap
    return text.strip()


def chunk_file(path: Path) -> tuple[dict | None, list[dict]]:
    """
    Process a single cleaned .md file.
    Return (doc_record, [chunk_records]); return (None, []) when no content.
    """
    raw  = path.read_text(encoding="utf-8")
    meta, body = _parse_frontmatter(raw)

    if not body.strip():
        return None, []

    source_url = meta.get("source_url", "")
    site_key   = meta.get("site_key", path.parent.name)
    category   = meta.get("category", "")
    doc_id     = meta.get("doc_id", f"D_{site_key}_{path.stem}")

    site_title = _SITE_TITLE_MAP.get(site_key) or _infer_site_title(body, site_key)
    _SITE_TITLE_MAP[site_key] = site_title

    sections = _split_sections(body)
    if not sections:
        return None, []

    # Document record
    doc = {
        "doc_id":      doc_id,
        "collection":  "D",
        "source":      re.sub(r"https?://(www\.)?", "", source_url).split("/")[0],
        "url":         source_url,
        "site_key":    site_key,
        "site_title":  site_title,
        "category":    category,
        "doc_type":    "webpage",
        "section_count": len(sections),
        "word_count":  _word_count(body),
        "scraped_at":  meta.get("crawled_at", ""),
        "cleaned_at":  meta.get("cleaned_at", ""),
    }

    # Chunk records
    chunks: list[dict] = []
    for i, sec in enumerate(sections):
        # Overlap: leading OVERLAP_WORDS from the next section
        next_body = sections[i + 1].body if i + 1 < len(sections) else ""
        overlap   = _overlap_text(_strip_md_syntax(next_body), OVERLAP_WORDS)

        text = _build_chunk_text(site_title, category, sec, overlap)

        breadcrumb_parts = sec.ancestors + ([sec.title] if sec.title else [])
        section_path     = " > ".join(breadcrumb_parts) if breadcrumb_parts else "(intro)"

        chunks.append({
            "chunk_id":    f"{doc_id}__{i:04d}",
            "doc_id":      doc_id,
            "collection":  "D",
            "source":      doc["source"],
            "url":         source_url,
            "site_key":    site_key,
            "site_title":  site_title,
            "category":    category,
            "doc_type":    "webpage",
            "section":     sec.title or "(intro)",
            "section_path": section_path,
            "chunk_index": i,
            "heading_level": sec.level,
            "word_count":  _word_count(sec.body),
            "has_overlap": bool(overlap),
            "text":        text,
        })

    return doc, chunks


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def chunk_all(
    site_filter: str = "",
    append: bool = True,
    dry_run: bool = False,
) -> tuple[list[dict], list[dict]]:
    """
    Process all .md files under data/processed/D/ and write D_docs.jsonl & D_chunks.jsonl.

    Parameters
    ----------
    site_filter : Only process files whose path contains this substring (empty = all)
    append      : True = append mode, False = overwrite
    dry_run     : Print stats only, do not write files
    """
    md_files = sorted(PROCESSED_D_DIR.rglob("*.md"))
    if site_filter:
        md_files = [f for f in md_files if site_filter.lower() in str(f).lower()]

    if not md_files:
        print(f"No .md files found under {PROCESSED_D_DIR}"
              + (f" matching '{site_filter}'" if site_filter else ""))
        return [], []

    print("=" * 60)
    print(f"Collection D — Chunking")
    print(f"{'DRY RUN  ' if dry_run else ''}Files: {len(md_files)}")
    print("=" * 60)

    all_docs:   list[dict] = []
    all_chunks: list[dict] = []
    site_stats: dict[str, dict] = {}

    for path in md_files:
        doc, chunks = chunk_file(path)
        rel = path.relative_to(PROCESSED_D_DIR)

        if doc is None:
            print(f"  skip  {rel}  (no content)")
            continue

        all_docs.append(doc)
        all_chunks.extend(chunks)

        sk = doc["site_key"]
        if sk not in site_stats:
            site_stats[sk] = {"files": 0, "chunks": 0, "words": 0}
        site_stats[sk]["files"]  += 1
        site_stats[sk]["chunks"] += len(chunks)
        site_stats[sk]["words"]  += doc["word_count"]

        print(f"  ok  {str(rel):<55} {len(chunks):>3} chunks")

    # Write files
    if not dry_run and all_docs:
        OUT_DOCS.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with open(OUT_DOCS, mode, encoding="utf-8") as f:
            for d in all_docs:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        with open(OUT_CHUNKS, mode, encoding="utf-8") as f:
            for c in all_chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # Summary
    print(f"\n{'='*60}")
    print("Per-site summary")
    print(f"{'='*60}")
    for sk, st in site_stats.items():
        print(f"  {sk:<30} {st['files']:>3} files  "
              f"{st['chunks']:>4} chunks  ~{st['words']:>6} words")

    total_chunks = len(all_chunks)
    total_words  = sum(d["word_count"] for d in all_docs)
    print(f"\n  {'Total':<30} {len(all_docs):>3} files  "
          f"{total_chunks:>4} chunks  ~{total_words:>6} words")

    if not dry_run and all_docs:
        docs_kb   = OUT_DOCS.stat().st_size   / 1024
        chunks_kb = OUT_CHUNKS.stat().st_size / 1024
        mode_str  = "Appended" if append else "Wrote"
        print(f"\n  {mode_str}:")
        print(f"    D_docs.jsonl   : {len(all_docs)} records  ({docs_kb:.1f} KB)")
        print(f"    D_chunks.jsonl : {total_chunks} records  ({chunks_kb:.1f} KB)")

    return all_docs, all_chunks


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Chunk Collection D cleaned markdown files"
    )
    parser.add_argument(
        "--site", type=str, default="",
        help="Only process files whose path contains this string"
    )
    parser.add_argument(
        "--no-append", action="store_true",
        help="Overwrite output files instead of appending"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print stats without writing output files"
    )
    args = parser.parse_args()

    chunk_all(
        site_filter=args.site,
        append=not args.no_append,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()