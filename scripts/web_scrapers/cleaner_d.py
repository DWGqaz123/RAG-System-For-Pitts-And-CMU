r"""
clean_collection_d.py
======================
Cleaning pipeline for Collection D markdown.

Steps:
    1) Keep YAML front matter and update cleaned_at
    2) Drop images ![]()
    3) Unwrap markdown links [text](url) -> text
    4) Drop reference link defs [ref]: url
    5) Drop reference links [text][ref] -> text
    6) Remove bare URLs
    7) Strip HTML tags
    8) Unescape HTML entities
    9) Remove markdown escape backslashes
    10) Remove nav/share noise lines
    11) Optionally drop isolated CTA button lines
    12) Collapse blank lines
    13) Trim line trailing spaces

Output: data/processed/D/<site_key>/<slug>.md with original front matter plus cleaned_at.
CLI: python clean_collection_d.py [--site FILTER] [--strip-buttons] [--dry-run]
"""

import argparse
import html
import re
from datetime import datetime, timezone
from pathlib import Path

# Paths

RAW_DIR  = Path("data/raw/D")
OUT_DIR  = Path("data/processed/D")

# Navigation / social noise filters (case-insensitive, whole-line match)

_NAV_PATTERNS = [
    r"skip\s+to\s+(main\s+)?content",
    r"back\s+to\s+top",
    r"jump\s+to\s+(navigation|content|section)",
    r"toggle\s+(navigation|menu|search)",
    r"main\s+navigation",
    r"breadcrumb",
    r"you\s+are\s+here",
    r"home\s*[>/»]",                    # breadcrumb fragment
    r"^\s*menu\s*$",
    r"^\s*navigation\s*$",
    r"^\s*close\s*$",
    r"^\s*open\s*$",
    r"^\s*search\s*$",
    r"cookie\s+(policy|notice|banner|consent|preferences)(?:\s+\w+)*",
    r"we\s+use\s+cookies(?:\s+.*)?",
    r"accept\s+(all\s+)?cookies",
    r"privacy\s+policy",               
    r"terms\s+(of\s+)?(use|service)",  
]

_SOCIAL_PATTERNS = [
    r"share\s+(on\s+)?(facebook|twitter|instagram|linkedin|x\.com)",
    r"follow\s+us\s+on",
    r"tweet\s+this",
    r"(like|share)\s+this\s+(page|post|article)",
    r"subscribe\s+to\s+our\s+(newsletter|email)",
    r"sign\s+up\s+for\s+(our\s+)?newsletter",
]

# 编译成单个正则（行级匹配）
_NOISE_RE = re.compile(
    r"^\s*(?:" + "|".join(_NAV_PATTERNS + _SOCIAL_PATTERNS) + r")\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# ---------------------------------------------------------------------------
# Standalone junk link detection
# Delete a line if it is just one markdown link and matches any of:
#   - link text all uppercase with letters
#   - link text starts with media terms (photo/image/logo/banner...)
#   - URL points to file extensions (pdf/doc/zip/mp4...)
#   - URL has junk patterns (wp-content/uploads, youtube channel, tracking)

_STANDALONE_LINK_RE = re.compile(
    r"^\s*\[(?P<text>[^\]]+)\]\((?P<url>[^)]+)\)\s*$"
)
_MEDIA_EXT_RE = re.compile(
    r"\.(pdf|doc|docx|xls|xlsx|ppt|pptx|zip|rar|mp4|mp3|wav|avi|mov|png|jpg|jpeg|gif|webp|svg)(\?|$)",
    re.IGNORECASE,
)
_MEDIA_TEXT_RE = re.compile(
    r"^(photo|image|img|picture|icon|logo|banner|thumbnail|video|audio)\b",
    re.IGNORECASE,
)
_JUNK_URL_RE = re.compile(
    r"(wp-content/uploads|youtube\.com/channel|embeds_referring|%2F.*%2F"
    r"|eapps\.|/forms?/|/applications?/|/permits?/|/contracts?/|/maps?/)",
    re.IGNORECASE,
)


def _is_junk_standalone_link(line: str) -> bool:
    """
    Detect if a line is just one markdown link that looks like junk (nav, media, tracking, etc.)."""
   
    m = _STANDALONE_LINK_RE.match(line)
    if not m:
        return False
    text = m.group("text").strip()
    url  = m.group("url").strip()
    if text == text.upper() and any(c.isalpha() for c in text):
        return True
    if _MEDIA_TEXT_RE.match(text):
        return True
    if _MEDIA_EXT_RE.search(url):
        return True
    if _JUNK_URL_RE.search(url):
        return True
    return False


# Optional CTA button blacklist (short standalone lines)

_BUTTON_WORDS = {
    "buy tickets", "get tickets", "learn more", "click here",
    "read more", "see more", "view more", "show more",
    "find out more", "discover more", "explore", "register now",
    "sign up", "log in", "login", "donate now", "donate",
    "shop now", "shop", "subscribe", "unsubscribe",
    "contact us", "get in touch", "book now", "reserve",
    "apply now", "apply", "download", "print",
    "add to cart", "checkout", "back", "next", "previous",
    "load more", "view all", "see all", "show all",
}


# Cleaning functions

def _strip_yaml(text: str) -> tuple[dict, str]:
    """Split YAML front matter and body. Return (meta_dict, body)."""
    if not text.startswith("---"):
        return {}, text

    end = text.find("\n---", 3)
    if end == -1:
        return {}, text

    yaml_block = text[3:end].strip()
    body       = text[end + 4:].lstrip("\n")

    meta = {}
    for line in yaml_block.splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            meta[k.strip()] = v.strip()

    return meta, body


def _build_yaml(meta: dict, cleaned_at: str) -> str:
    """Serialize meta back to YAML front matter string."""
    meta["cleaned_at"] = cleaned_at
    lines = ["---"]
    for k, v in meta.items():
        lines.append(f"{k}: {v}")
    lines.append("---\n")
    return "\n".join(lines)


def clean_body(text: str, strip_button_lines: bool = False) -> str:
    """Run all cleaning steps on markdown body and return cleaned text."""
    lines_0 = text.splitlines()
    text = "\n".join("" if _is_junk_standalone_link(ln) else ln for ln in lines_0)

    text = re.sub(r"!\[(?:[^\[\]]*)\]\([^)]*\)", "", text)

    for _ in range(5):
        new = re.sub(r"\[([^\[\]]*)\]\([^)]*\)", r"\1", text)
        if new == text:
            break
        text = new

    text = re.sub(
        r"^\s*\[[^\]]+\]:\s+\S+(?:\s+[\"(][^\")]*[\")'])?\s*$",
        "",
        text,
        flags=re.MULTILINE,
    )

    text = re.sub(r"\[([^\[\]]+)\]\[[^\]]*\]", r"\1", text)

    text = re.sub(r"https?://\S+", "", text)

    text = re.sub(r"<[^>]+>", "", text)

    text = html.unescape(text)
    text = re.sub(r"&nbsp;", " ", text)    # html.unescape 不处理 &nbsp;

    text = re.sub(r"\\([\\`*_{}\[\]()#+\-.!|>~])", r"\1", text)

    text = _NOISE_RE.sub("", text)


    if strip_button_lines:
        def _is_button_line(line: str) -> bool:
            stripped = line.strip().rstrip(".!").lower()
            return stripped in _BUTTON_WORDS

        lines = text.splitlines()
        text  = "\n".join(
            "" if _is_button_line(ln) else ln
            for ln in lines
        )

    text = re.sub(r"\n{3,}", "\n\n", text)

    lines = [ln.rstrip() for ln in text.splitlines()]
    text  = "\n".join(lines)

    return text.strip()


# File processing

def clean_file(
    src: Path,
    dst: Path,
    strip_button_lines: bool = False,
    dry_run: bool = False,
) -> dict:
    """Clean one markdown file and return stats."""
    raw  = src.read_text(encoding="utf-8")
    meta, body = _strip_yaml(raw)

    before_chars = len(body)
    cleaned_body = clean_body(body, strip_button_lines=strip_button_lines)
    after_chars  = len(cleaned_body)

    if not dry_run:
        dst.parent.mkdir(parents=True, exist_ok=True)
        ts      = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        content = _build_yaml(meta, ts) + "\n" + cleaned_body + "\n"
        dst.write_text(content, encoding="utf-8")

    return {
        "src":    src,
        "dst":    dst,
        "before": before_chars,
        "after":  after_chars,
        "pct":    round(100 * (1 - after_chars / before_chars), 1) if before_chars else 0,
    }


# Batch processing

def clean_all(
    site_filter: str = "",
    strip_button_lines: bool = False,
    dry_run: bool = False,
) -> list[dict]:
    """Clean all .md under data/raw/D, writing to data/processed/D/."""
    src_files = sorted(RAW_DIR.rglob("*.md"))
    if site_filter:
        src_files = [f for f in src_files if site_filter.lower() in str(f).lower()]

    if not src_files:
        print(f"No .md files found under {RAW_DIR}"
              + (f" matching '{site_filter}'" if site_filter else ""))
        return []

    print("=" * 60)
    print(f"Collection D — Markdown Cleaning")
    print(f"{'DRY RUN  ' if dry_run else ''}Files: {len(src_files)}")
    print("=" * 60)

    stats = []
    site_totals: dict[str, list] = {}

    for src in src_files:
        rel = src.relative_to(RAW_DIR)
        dst = OUT_DIR / rel

        stat = clean_file(src, dst, strip_button_lines=strip_button_lines, dry_run=dry_run)
        stats.append(stat)

        site = rel.parts[0] if len(rel.parts) > 1 else "root"
        site_totals.setdefault(site, []).append(stat)

        print(f"  {'(dry)' if dry_run else 'ok  '} "
              f"{str(rel):<60} "
              f"{stat['before']:>7} → {stat['after']:>7} chars  "
              f"(-{stat['pct']}%)")

    print(f"\n{'='*60}")
    print("Per-site summary")
    print(f"{'='*60}")
    total_before = total_after = 0
    for site, site_stats in site_totals.items():
        b = sum(s["before"] for s in site_stats)
        a = sum(s["after"]  for s in site_stats)
        p = round(100 * (1 - a / b), 1) if b else 0
        total_before += b
        total_after  += a
        print(f"  {site:<30} {len(site_stats):>3} files  "
              f"{b:>8} → {a:>8} chars  (-{p}%)")

    overall = round(100 * (1 - total_after / total_before), 1) if total_before else 0
    print(f"\n  {'Total':<30} {len(stats):>3} files  "
          f"{total_before:>8} → {total_after:>8} chars  (-{overall}%)")
    if not dry_run:
        print(f"\n  Output: {OUT_DIR.resolve()}")

    return stats


# CLI

def main():
    parser = argparse.ArgumentParser(
        description="Clean Collection D FireCrawl markdown files"
    )
    parser.add_argument(
        "--site", type=str, default="",
        help="Only process files whose path contains this string"
    )
    parser.add_argument(
        "--strip-buttons", action="store_true",
        help="Also remove isolated CTA button lines (Learn More, Buy Tickets, etc.)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print stats without writing output files"
    )
    args = parser.parse_args()

    clean_all(
        site_filter=args.site,
        strip_button_lines=args.strip_buttons,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()