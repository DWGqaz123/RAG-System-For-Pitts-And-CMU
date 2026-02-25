"""
chunker_collection_a.py
----------
Chunker for RAG Knowledge Base (Collection A - heading-aware).

Input:  JSONL docs where each line is a dict with fields like:
  {
    "doc_id": "...",
    "collection": "A",
    "source": "...",
    "url": "...",
    "title": "...",
    "fetched_at": "...",
    "text": "markdown body (YAML removed)"
  }

Output: JSONL chunks where each line is a chunk dict:
  {
    "chunk_id": "...",
    "doc_id": "...",
    "collection": "A",
    "source": "...",
    "url": "...",
    "title": "...",
    "md_title": "...",
    "section": "...",
    "subsection": "... or null",
    "chunk_index": 0,
    "text": "Title: ...\nSection: ...\nSubsection: ...\nContent:\n..."
  }

Chunking logic (per spec):
  - '#'  (H1) is stored as md_title metadata (no separate chunk).
  - Content before the first '##' is treated as a special intro section.
  - Each '##' starts a section.
  - If a section contains '###', split by '###' (intro chunk + each subsection chunk).
  - If a chunk is too long:
      1. Split by paragraphs '\n\n' and pack to target length.
      2. If a single paragraph is still too long, split by sentences (. ? !).
      3. Final fallback: fixed window with overlap.

Deterministic — no external models required.

Usage (import in notebook):
    from chunker import chunk_collection
    chunks = chunk_collection("A")
"""

import json
import re
import argparse
from pathlib import Path
from urllib.parse import urlparse
import html


# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

PROCESSED_DIR = Path("data/processed")
RAW_WIKI_DIR  = Path("data/raw/A/wikipedias_base_line")
DOCS_PATH     = PROCESSED_DIR / "A_docs.jsonl"
CHUNKS_PATH   = PROCESSED_DIR / "A_chunks.jsonl"

TARGET_CHARS   = 1000   # Soft target chunk size in characters
MAX_CHARS      = 1500   # Hard ceiling before overflow splitting kicks in
OVERLAP_CHARS  = 150    # Overlap for fixed-window fallback


# ──────────────────────────────────────────────
# Text splitting utilities
# ──────────────────────────────────────────────

def _split_sentences(text: str) -> list[str]:
    """
    Split text into sentences on '.', '?', '!' boundaries,
    keeping the delimiter attached to the preceding sentence.
    """
    parts = re.split(r'(?<=[.?!])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def _pack(units: list[str], target: int, max_size: int) -> list[str]:
    """
    Greedily pack units (paragraphs or sentences) into chunks no larger
    than max_size, aiming for target size.  A single oversized unit is
    emitted as its own chunk rather than being silently dropped.
    """
    chunks: list[str] = []
    current_parts: list[str] = []
    current_len = 0

    for unit in units:
        unit_len = len(unit)
        join_len = current_len + (2 if current_parts else 0) + unit_len

        if current_parts and join_len > max_size:
            chunks.append("\n\n".join(current_parts))
            current_parts = [unit]
            current_len = unit_len
        else:
            current_parts.append(unit)
            current_len = join_len

    if current_parts:
        chunks.append("\n\n".join(current_parts))

    return chunks


def _fixed_window(text: str, target: int, overlap: int) -> list[str]:
    """
    Last-resort fixed-window split with character overlap.
    Applied when a single sentence still exceeds max_size.
    """
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + target
        chunks.append(text[start:end].strip())
        start += target - overlap
    return [c for c in chunks if c]


def split_text(text: str, target: int = TARGET_CHARS, max_size: int = MAX_CHARS,
               overlap: int = OVERLAP_CHARS) -> list[str]:
    """
    Split a block of text into chunks using a three-tier cascade:
      1. Pack paragraphs up to max_size.
      2. For oversized paragraphs, pack sentences.
      3. For oversized sentences, use fixed window with overlap.
    """
    if len(text) <= max_size:
        return [text]

    # Tier 1: paragraph-level packing
    paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
    chunks: list[str] = []

    for para in paragraphs:
        if len(para) <= max_size:
            chunks.append(para)
        else:
            # Tier 2: sentence-level packing
            sentences = _split_sentences(para)
            sent_chunks = _pack(sentences, target, max_size)

            for sc in sent_chunks:
                if len(sc) <= max_size:
                    chunks.append(sc)
                else:
                    # Tier 3: fixed window
                    chunks.extend(_fixed_window(sc, target, overlap))

    return _pack(chunks, target, max_size)


# ──────────────────────────────────────────────
# Shared helpers for wiki processing
# ──────────────────────────────────────────────

_IMG_RE      = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK_RE     = re.compile(r"\[([^\[\]]+)\]\([^)]*\)")
_REF_DEF_RE  = re.compile(r"^\s*\[[^\]]+\]:\s+\S.*$", re.MULTILINE)
_REF_LINK_RE = re.compile(r"\[([^\[\]]+)\]\[[^\]]*\]")
_BARE_URL_RE = re.compile(r"https?://\S+")
_ENTITY_RE   = re.compile(r"&([a-zA-Z]{2,8}|#\d{1,5}|#x[0-9a-fA-F]{1,5});")

_YAML_FENCE = re.compile(r"^---\n(.*?)\n---\n?", re.DOTALL)

_HTML_ENTITIES = {
    "amp": "&", "lt": "<", "gt": ">", "quot": '"',
    "apos": "'", "nbsp": " ", "ndash": "–", "mdash": "—",
    "lsquo": "'", "rsquo": "'", "ldquo": "“", "rdquo": "”",
    "hellip": "…", "bull": "•", "copy": "©", "reg": "®",
}


def _decode_entity(m: re.Match) -> str:
    name = m.group(1)
    if name.startswith("#x"):
        return chr(int(name[2:], 16))
    if name.startswith("#"):
        return chr(int(name[1:]))
    return _HTML_ENTITIES.get(name.lower(), m.group(0))


def _parse_frontmatter(raw: str) -> tuple[dict, str]:
    m = _YAML_FENCE.match(raw)
    if not m:
        return {}, raw
    meta: dict[str, str] = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            meta[k.strip()] = v.strip()
    return meta, raw[m.end():]


def clean_wiki_markdown(text: str) -> str:
    """Light cleaning: strip links/urls/entities, keep anchor text."""
    text = _IMG_RE.sub("", text)
    for _ in range(5):
        t2 = _LINK_RE.sub(r"\1", text)
        if t2 == text:
            break
        text = t2
    text = _REF_DEF_RE.sub("", text)
    text = _REF_LINK_RE.sub(r"\1", text)
    text = _BARE_URL_RE.sub("", text)
    text = _ENTITY_RE.sub(_decode_entity, text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_source(url: str) -> str:
    if not url:
        return ""
    host = urlparse(url).netloc
    return re.sub(r"^www\.", "", host)


# ──────────────────────────────────────────────
# Markdown heading parser
# ──────────────────────────────────────────────

def _heading_level(line: str) -> int | None:
    """Return heading level (1-3) if line is a heading, else None."""
    m = re.match(r'^(#{1,3})\s+', line)
    return len(m.group(1)) if m else None


def _heading_text(line: str) -> str:
    """Strip leading #s and whitespace from a heading line."""
    return re.sub(r'^#+\s+', '', line).strip()


def parse_sections(text: str) -> dict:
    """
    Parse a Markdown document body into a structured dict.

    H1 handling:
      - The FIRST H1 becomes md_title (document-level metadata).
      - SUBSEQUENT H1s are treated as top-level section boundaries
        (equivalent to H2), with their text stored in the section's
        'h1_title' field.  This handles multi-topic pages (e.g. Britannica)
        where each major subject starts with its own H1.

    Return schema:
    {
      "md_title": str | None,
      "sections": [
        {
          "heading":    str,          # H2 text, or H1 text for subsequent H1s,
                                      # or "__intro__" for pre-heading content
          "h1_title":   str | None,   # Set only when section was opened by a
                                      # subsequent H1 (not a real H2)
          "intro":      str,          # body before first H3 in this section
          "subsections": [
            {"heading": str, "body": str},
            ...
          ]
        },
        ...
      ]
    }
    """
    lines = text.splitlines()

    md_title: str | None = None
    sections: list[dict] = []

    current_section: dict | None = None
    current_sub_heading: str | None = None
    current_sub_lines: list[str] = []
    current_intro_lines: list[str] = []
    pre_h2_lines: list[str] = []

    def flush_subsection():
        if current_sub_heading is not None and current_section is not None:
            current_section["subsections"].append({
                "heading": current_sub_heading,
                "body": "\n".join(current_sub_lines).strip(),
            })

    def flush_section():
        nonlocal current_sub_heading, current_sub_lines, current_intro_lines
        if current_section is not None:
            flush_subsection()
            current_section["intro"] = "\n".join(current_intro_lines).strip()
            sections.append(current_section)
        current_sub_heading = None
        current_sub_lines = []
        current_intro_lines = []

    in_section = False          # True once we have entered any H1/H2 block
    pre_h2_flushed = False      # Guard: emit pre-section block at most once
    active_h1_title: str | None = None   # Tracks the most recent subsequent H1

    def flush_pre_h2():
        """Emit accumulated pre-section lines as '__intro__' (once only)."""
        nonlocal pre_h2_flushed
        if pre_h2_flushed:
            return
        pre_h2_flushed = True
        body = "\n".join(pre_h2_lines).strip()
        if body:
            sections.append({
                "heading": "__intro__",
                "h1_title": None,
                "intro": body,
                "subsections": [],
            })

    def open_section(heading: str, from_h1: bool = False):
        """Open a new section, flushing the previous one first."""
        nonlocal in_section, current_section, active_h1_title
        if not in_section:
            flush_pre_h2()
        flush_section()
        in_section = True
        # A subsequent H1 sets a new H1 context; a real H2 inherits the current one
        if from_h1:
            active_h1_title = heading
        current_section = {
            "heading":     heading,
            "h1_title":    active_h1_title,
            "intro":       "",
            "subsections": [],
        }

    for line in lines:
        level = _heading_level(line)

        if level == 1:
            heading_text = _heading_text(line)
            if md_title is None:
                # First H1 -> document-level title, no section opened
                md_title = heading_text
            else:
                # Subsequent H1 -> treat as a new top-level section
                open_section(heading_text, from_h1=True)
            continue

        if level == 2:
            open_section(_heading_text(line), from_h1=False)
            continue

        if level == 3:
            if current_section is None:
                # H3 before any H1/H2 -- open an implicit intro section first
                flush_pre_h2()
                current_section = {
                    "heading": "__intro__",
                    "h1_title": None,
                    "intro": "",
                    "subsections": [],
                }
                in_section = True
            flush_subsection()
            current_sub_heading = _heading_text(line)
            current_sub_lines = []
            continue

        # Regular content line
        if not in_section:
            pre_h2_lines.append(line)
        elif current_sub_heading is not None:
            current_sub_lines.append(line)
        else:
            current_intro_lines.append(line)

    # Flush whatever is still open
    flush_section()

    # Documents with no headings at all, or only pre-section content
    if not pre_h2_flushed:
        flush_pre_h2()

    return {"md_title": md_title, "sections": sections}


# ──────────────────────────────────────────────
# Build docs from raw Wikipedia markdown
# ──────────────────────────────────────────────

def _build_doc_from_wiki(path: Path) -> dict | None:
    raw = path.read_text(encoding="utf-8")
    meta, body = _parse_frontmatter(raw)
    body = clean_wiki_markdown(body)
    if not body.strip():
        return None

    url = meta.get("source_url", "")
    doc_id = meta.get("doc_id") or f"A_wiki_{path.stem}"
    title = meta.get("title") or path.stem.replace("_", " ").title()

    return {
        "doc_id":     doc_id,
        "collection": "A",
        "source":     extract_source(url),
        "url":        url,
        "title":      title,
        "fetched_at": meta.get("scraped_at", ""),
        "text":       body,
    }


def build_docs_from_wiki(file_filter: str = "", dry_run: bool = False,
                         append: bool = False) -> list[dict]:
    md_files = sorted(RAW_WIKI_DIR.rglob("*.md"))
    if file_filter:
        md_files = [f for f in md_files if file_filter.lower() in f.name.lower()]

    if not md_files:
        print(f"No .md files found under {RAW_WIKI_DIR}" + (f" matching '{file_filter}'" if file_filter else ""))
        return []

    docs: list[dict] = []
    for path in md_files:
        doc = _build_doc_from_wiki(path)
        if doc is None:
            print(f"  skip  {path.name} (empty after cleaning)")
            continue
        docs.append(doc)
        print(f"  ok    {path.name:<40} {len(doc['text'].split()):>5} words")

    if dry_run or not docs:
        return docs

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with open(DOCS_PATH, mode, encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"  wrote {len(docs)} docs to {DOCS_PATH}")
    return docs


# ──────────────────────────────────────────────
# Chunk builder
# ──────────────────────────────────────────────

def _format_chunk_text(title: str, section: str, subsection: str | None,
                        content: str, h1_title: str | None = None) -> str:
    """Build the final text string that will be embedded / retrieved."""
    lines = [f"Title: {title}"]
    # For sections opened by a subsequent H1, surface it as its own labelled line
    if h1_title:
        lines.append(f"H1 Section: {h1_title}")
    lines.append(f"Section: {section}")
    if subsection:
        lines.append(f"Subsection: {subsection}")
    lines += ["Content:", content]
    return "\n".join(lines)


def build_chunks(doc: dict) -> list[dict]:
    """
    Produce all chunks for a single document dict.

    Chunk ID format: <doc_id>__<zero-padded-index>
    """
    doc_id     = doc["doc_id"]
    collection = doc["collection"]
    source     = doc["source"]
    url        = doc["url"]
    title      = doc["title"]
    text       = doc.get("text", "")

    parsed   = parse_sections(text)
    md_title = parsed["md_title"] or title

    chunks: list[dict] = []

    def emit(section_heading: str, subsection_heading: str | None, content: str,
              h1_title: str | None = None):
        """Split content if needed and append chunk records."""
        content = content.strip()
        if not content:
            return

        # Use doc title as section label for intro-only docs
        display_section = title if section_heading == "__intro__" else section_heading

        blocks = split_text(content)
        for block in blocks:
            block = block.strip()
            if not block:
                continue
            idx = len(chunks)
            chunks.append({
                "chunk_id":    f"{doc_id}__{idx:04d}",
                "doc_id":      doc_id,
                "collection":  collection,
                "source":      source,
                "url":         url,
                "title":       title,
                "md_title":    md_title,
                "section":     display_section,
                "subsection":  subsection_heading,
                "chunk_index": idx,
                "text": _format_chunk_text(title, display_section,
                                           subsection_heading, block,
                                           h1_title=h1_title),
            })

    for sec in parsed["sections"]:
        sec_heading = sec["heading"]
        h1_title    = sec.get("h1_title")   # Non-null only for subsequent-H1 sections

        # Section intro (content before first H3)
        if sec["intro"]:
            emit(sec_heading, None, sec["intro"], h1_title)

        # Each H3 subsection
        for sub in sec["subsections"]:
            emit(sec_heading, sub["heading"], sub["body"], h1_title)

    return chunks


# ──────────────────────────────────────────────
# Collection-level entry point
# ──────────────────────────────────────────────

def chunk_collection(collection: str,
                     input_path: Path | None = None,
                     output_path: Path | None = None) -> list[dict]:
    """
    Read <collection>_docs.jsonl, chunk every document, and write
    <collection>_chunks.jsonl to the same directory.

    Args:
        collection:   Single uppercase letter, e.g. "A".
        input_path:   Override default input path (optional).
        output_path:  Override default output path (optional).

    Returns:
        List of all chunk dicts (also written to JSONL).
    """
    input_path  = input_path  or PROCESSED_DIR / f"{collection}_docs.jsonl"
    output_path = output_path or PROCESSED_DIR / f"{collection}_chunks.jsonl"

    if not input_path.exists():
        raise FileNotFoundError(
            f"{input_path} not found. Generate {collection}_docs.jsonl before chunking."
        )

    docs = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines()
            if line.strip()]

    return chunk_docs(docs, output_path)


def chunk_docs(docs: list[dict], output_path: Path | None = None) -> list[dict]:
    """Chunk an in-memory list of docs and optionally write JSONL."""
    output_path = output_path or CHUNKS_PATH

    all_chunks: list[dict] = []
    for doc in docs:
        doc_chunks = build_chunks(doc)
        all_chunks.extend(doc_chunks)
        print(f"  ✓  {doc['doc_id']:<45}  {len(doc_chunks):>3} chunks")

    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    size_kb = output_path.stat().st_size / 1024
    print(f"\n  → Wrote {len(all_chunks)} chunks to {output_path}  ({size_kb:.1f} KB)")
    return all_chunks


def run_wiki_pipeline(file_filter: str = "", append: bool = False,
                      dry_run: bool = False) -> tuple[list[dict], list[dict]]:
    """Clean raw wiki markdown, build docs, then chunk them."""
    docs = build_docs_from_wiki(file_filter=file_filter, dry_run=dry_run, append=append)
    if dry_run or not docs:
        return docs, []
    chunks = chunk_docs(docs, output_path=CHUNKS_PATH)
    return docs, chunks


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk Collection A data")
    parser.add_argument(
        "--from-wiki", action="store_true",
        help="Clean raw wiki markdown then chunk (data/raw/A/wikipedias_base_line)"
    )
    parser.add_argument(
        "--file", type=str, default="",
        help="Filter raw wiki files by substring (only with --from-wiki)"
    )
    parser.add_argument(
        "--append", action="store_true",
        help="Append when writing docs (only with --from-wiki)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print stats without writing output files"
    )
    args = parser.parse_args()

    if args.from_wiki:
        run_wiki_pipeline(file_filter=args.file, append=args.append, dry_run=args.dry_run)
    else:
        chunk_collection("A")


if __name__ == "__main__":
    main()