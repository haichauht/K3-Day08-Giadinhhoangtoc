"""Standardize landing documents as Markdown for the RAG pipeline."""

from __future__ import annotations

import argparse
import io
import json
import re
from pathlib import Path
from urllib.parse import urlparse

from markitdown import MarkItDown

REPO_ROOT = Path(__file__).resolve().parent.parent
LANDING_DIR = REPO_ROOT / "data" / "landing"
OUTPUT_DIR = REPO_ROOT / "data" / "standardized"

# This PDF has a broken legacy-font text layer. Its standardized Markdown was
# transcribed from rendered pages and must not be overwritten by MarkItDown.
MANUAL_LEGAL_STEMS = {"quy-dinh-dao-tao-song-nganh-dhqg-hcm"}


def yaml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def load_legal_sources(legal_dir: Path) -> dict[str, dict]:
    source_path = legal_dir / "sources.json"
    if not source_path.exists():
        return {}
    sources = json.loads(source_path.read_text(encoding="utf-8"))
    return {item["filename"]: item for item in sources}


def legal_front_matter(filepath: Path, metadata: dict) -> str:
    title = metadata.get("title", filepath.stem.replace("-", " ").title())
    downloaded_at = metadata.get("downloaded_at", "")
    retrieved_at = downloaded_at[:10] if downloaded_at else "not-stated"
    values = [
        "---",
        f"doc_id: {yaml_string(f'legal-{filepath.stem}')}",
        f"title: {yaml_string(title)}",
        f"source_url: {yaml_string(metadata.get('url', 'not-stated'))}",
        f"source_file: {yaml_string(filepath.relative_to(REPO_ROOT).as_posix())}",
        f"retrieved_at: {yaml_string(retrieved_at)}",
        'document_version: "not-stated"',
        'page_published_at: "not-stated"',
        'audience: "student"',
        'institution: "uit-dhqg-hcm"',
        'category: "academic-regulation"',
        'language: "vi"',
        'conversion_method: "markitdown-pdf+rule-based-structure"',
        'cleaning_version: "markitdown-structured-v2"',
        "---",
    ]
    return "\n".join(values), title


def structure_legal_markdown(text: str) -> str:
    """Promote Vietnamese legal sections to headings and remove duplicate TOCs."""
    lines = [line.strip() for line in text.replace("\r\n", "\n").split("\n")]
    output: list[str] = []
    in_toc = False
    in_decision_articles = False
    skip_next = False

    def append_block(value: str) -> None:
        if output and output[-1] != "":
            output.append("")
        output.append(value)
        output.append("")

    for index, line in enumerate(lines):
        if skip_next:
            skip_next = False
            continue
        if not line:
            if output and output[-1] != "":
                output.append("")
            continue
        if re.fullmatch(r"\d+", line):
            continue

        if line.upper() == "MỤC LỤC":
            in_toc = True
            continue
        if in_toc:
            if line.upper() == "DANH MỤC TỪ VIẾT TẮT":
                in_toc = False
                append_block("## Danh mục từ viết tắt")
            continue

        # A chapter is sometimes split across two lines: "Chương I" followed
        # by its uppercase title. Combine them into one useful RAG heading.
        if re.fullmatch(r"Chương\s+[IVXLCDM]+", line, flags=re.IGNORECASE):
            next_line = lines[index + 1] if index + 1 < len(lines) else ""
            if (
                next_line
                and next_line == next_line.upper()
                and not re.search(r"\.{3,}", next_line)
            ):
                append_block(f"## {line}. {next_line}")
                skip_next = True
            else:
                append_block(f"## {line}")
            continue

        if re.match(r"^CHƯƠNG\s+(?:\d+|[IVXLCDM]+)\b", line):
            if not re.search(r"\.{3,}", line):
                append_block(f"## {line}")
            continue

        article_match = re.match(r"^Điều\s+(\d+)[.:]\s*(.*)$", line)
        if article_match and not re.search(r"\.{3,}", line):
            if in_decision_articles:
                append_block(f"### Điều {article_match.group(1)}")
                if article_match.group(2):
                    output.append(article_match.group(2))
            else:
                append_block(f"### {line}")
            continue

        special_headings = {
            "QUYẾT ĐỊNH": "Quyết định ban hành",
            "QUYẾT ĐỊNH:": "Nội dung quyết định",
            "QUY CHẾ": "Quy chế",
            "QUY ĐỊNH": "Quy định",
            "DANH MỤC TỪ VIẾT TẮT": "Danh mục từ viết tắt",
        }
        if line.upper() in {
            "QUYẾT ĐỊNH",
            "QUYẾT ĐỊNH:",
            "QUY CHẾ",
            "QUY ĐỊNH",
            "DANH MỤC TỪ VIẾT TẮT",
        }:
            key = line.upper()
            append_block(f"## {special_headings[key]}")
            if key == "QUYẾT ĐỊNH:":
                in_decision_articles = True
            elif key in {"QUY CHẾ", "QUY ĐỊNH"}:
                in_decision_articles = False
            continue

        if re.match(r"^(PHỤ LỤC|Phụ lục)\s+[IVXLCDM\d]+", line):
            append_block(f"## {line}")
            continue

        output.append(line)

    # Collapse excessive blank lines while retaining Markdown block spacing.
    cleaned: list[str] = []
    for line in output:
        if line or not cleaned or cleaned[-1]:
            cleaned.append(line)
    return "\n".join(cleaned).strip()


def news_slug(source_url: str) -> str:
    slug = Path(urlparse(source_url).path).name.casefold()
    return re.sub(r"[^a-z0-9-]+", "-", slug).strip("-") or "article"


def news_front_matter(payload: dict, slug: str) -> str:
    values = [
        "---",
        f"doc_id: {yaml_string(f'uit-news-{slug}')}",
        f"title: {yaml_string(payload['title'])}",
        f"source_url: {yaml_string(payload['url'])}",
        'source_section: "Tin UIT"',
        f"retrieved_at: {yaml_string(payload['retrieved_at'])}",
        'document_version: "not-stated"',
        f"page_published_at: {yaml_string(payload.get('published_at', 'not-stated'))}",
        'audience: "public"',
        'institution: "uit"',
        'department: "communications"',
        'category: "news"',
        'language: "vi"',
        'cleaning_version: "markitdown-html-v2"',
        "---",
    ]
    return "\n".join(values)


def convert_legal_docs() -> tuple[int, int]:
    """Convert legal PDFs with good text layers and preserve manual overrides."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = load_legal_sources(legal_dir)
    converter = MarkItDown(enable_plugins=False)
    converted = 0
    preserved = 0

    for filepath in sorted(legal_dir.rglob("*.pdf")):
        relative = filepath.relative_to(legal_dir).with_suffix(".md")
        output_path = output_dir / relative
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if filepath.stem in MANUAL_LEGAL_STEMS:
            if not output_path.exists():
                raise FileNotFoundError(
                    f"Missing manual standardized file: {output_path}"
                )
            print(f"Preserving manual transcription: {output_path.name}")
            preserved += 1
            continue

        print(f"Converting with MarkItDown: {filepath.name}")
        result = converter.convert(filepath)
        body = structure_legal_markdown(result.text_content)
        if len(body) < 200:
            raise ValueError(f"Converted content is unexpectedly short: {filepath}")
        header, title = legal_front_matter(filepath, sources.get(filepath.name, {}))
        output_path.write_text(f"{header}\n\n# {title}\n\n{body}\n", encoding="utf-8")
        print(f"  Saved: {output_path}")
        converted += 1

    return converted, preserved


def standardize_news() -> tuple[int, int]:
    """Convert raw landing JSON snapshots into standardized Markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)
    converter = MarkItDown(enable_plugins=False)
    converted = 0
    preserved = 0
    if not news_dir.exists():
        return converted, preserved

    for filepath in sorted(news_dir.glob("*.json")):
        payload = json.loads(filepath.read_text(encoding="utf-8"))
        required = ("url", "title", "retrieved_at")
        missing = [field for field in required if not payload.get(field)]
        if missing:
            raise ValueError(f"{filepath} is missing fields: {', '.join(missing)}")

        slug = news_slug(payload["url"])
        output_path = output_dir / f"{slug}.md"
        raw_html = str(payload.get("content_html", "")).strip()
        if raw_html:
            result = converter.convert_stream(
                io.BytesIO(raw_html.encode("utf-8")),
                file_extension=".html",
                url=payload["url"],
            )
            body = result.text_content.strip()
        else:
            # The five retained snapshots predate raw-HTML storage. Their richer
            # standardized files are authoritative and must not be downgraded to
            # the short inspection text kept in the legacy JSON.
            if output_path.exists():
                preserved += 1
                continue
            body = str(
                payload.get("content_markdown") or payload.get("content") or ""
            ).strip()

        if len(body) < 100:
            raise ValueError(f"News content is unexpectedly short: {filepath}")
        header = news_front_matter(payload, slug)
        output_path.write_text(
            f"{header}\n\n# {payload['title']}\n\n{body}\n",
            encoding="utf-8",
        )
        converted += 1
    return converted, preserved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standardize landing data")
    parser.add_argument(
        "--legal-only",
        action="store_true",
        help="Only convert legal PDFs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    converted, preserved = convert_legal_docs()
    print(f"Legal: converted={converted}, manual={preserved}")
    if not args.legal_only:
        news_converted, news_preserved = standardize_news()
        print(f"News: converted={news_converted}, preserved={news_preserved}")
    print(f"Done: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
