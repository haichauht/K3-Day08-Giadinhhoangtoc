"""
Task 3 — Convert toàn bộ file trong data/landing/ thành Markdown.

Sử dụng MarkItDown của Microsoft:
    https://github.com/microsoft/markitdown

Cài đặt:
    pip install "markitdown[pdf]"
    # Lưu ý: cần extra [pdf] để convert được file PDF. Chỉ "pip install markitdown"
    # (không có extra) sẽ báo MissingDependencyException khi convert PDF, dù JSON/DOCX
    # vẫn convert bình thường.

Hướng dẫn:
    1. Scan toàn bộ file trong data/landing/ (PDF, DOCX, JSON)
    2. Convert sang Markdown
    3. Lưu vào data/standardized/ giữ nguyên cấu trúc thư mục
"""

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from markitdown import MarkItDown

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"
LEGAL_EXTENSIONS = {".pdf", ".docx", ".doc"}
PDF_TEMP_DIR = Path(__file__).parent.parent / "tmp" / "pdfs"
VIETNAMESE_DIACRITICS = set(
    "ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩị"
    "óòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"
    "ĂÂĐÊÔƠƯÁÀẢÃẠẤẦẨẪẬẮẰẲẴẶÉÈẺẼẸẾỀỂỄỆÍÌỈĨỊ"
    "ÓÒỎÕỌỐỒỔỖỘỚỜỞỠỢÚÙỦŨỤỨỪỬỮỰÝỲỶỸỴ"
)


def _load_legal_manifest() -> dict[str, dict]:
    """Load legal source metadata, indexed by the downloaded filename."""
    manifest_path = LANDING_DIR / "legal" / "sources.json"
    if not manifest_path.exists():
        return {}

    items = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(items, list):
        raise ValueError(f"Legal manifest must contain a JSON list: {manifest_path}")

    return {
        item["filename"]: item
        for item in items
        if isinstance(item, dict) and "filename" in item
    }


def _looks_like_broken_vietnamese(text: str) -> bool:
    """Detect a likely broken text layer in a Vietnamese document."""
    letter_count = sum(character.isalpha() for character in text)
    if letter_count < 1_000:
        return False

    diacritic_count = sum(character in VIETNAMESE_DIACRITICS for character in text)
    return diacritic_count / letter_count < 0.01


def _ocr_pdf(pdf_path: Path) -> str | None:
    """OCR a PDF with local Poppler and Tesseract, if both are available."""
    pdftoppm = shutil.which("pdftoppm")
    tesseract = shutil.which("tesseract")
    if not pdftoppm or not tesseract:
        print("  [WARN] OCR skipped: pdftoppm or tesseract is unavailable")
        return None
    PDF_TEMP_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(
            prefix=f"ocr-{pdf_path.stem}-",
            dir=PDF_TEMP_DIR,
        ) as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            page_prefix = temp_dir / "page"
            subprocess.run(
                [
                    pdftoppm,
                    "-r",
                    "220",
                    "-png",
                    str(pdf_path),
                    str(page_prefix),
                ],
                check=True,
                capture_output=True,
            )

            page_images = sorted(
                temp_dir.glob("page-*.png"),
                key=lambda path: int(path.stem.rsplit("-", 1)[-1]),
            )
            if not page_images:
                print(f"  [WARN] OCR renderer produced no pages: {pdf_path}")
                return None

            pages = []
            for page_number, image_path in enumerate(page_images, start=1):
                result = subprocess.run(
                    [
                        tesseract,
                        str(image_path),
                        "stdout",
                        "-l",
                        "vie",
                        "--psm",
                        "3",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                page_text = result.stdout.strip()
                if page_text:
                    pages.append(f"## Trang {page_number}\n\n{page_text}")

            return "\n\n---\n\n".join(pages) or None
    except (OSError, subprocess.CalledProcessError) as error:
        print(f"  [WARN] OCR failed for {pdf_path.name}: {error}")
        return None


def _structure_legal_markdown(text: str) -> str:
    """Promote legal chapter/article markers to reproducible Markdown headings."""
    chapter_pattern = re.compile(
        r"^\s*((?i:chương)\s+(?:[IVXLCDMJI]+|\d+)\b.*)$"
    )
    article_pattern = re.compile(
        r"^\s*((?i:điều)\s+\d+[a-zA-Z]?(?:[.:])?.*)$"
    )
    table_of_contents_pattern = re.compile(r"\.{5,}\s*\d*\s*$")

    structured_lines: list[str] = []
    for original_line in text.splitlines():
        line = original_line.strip()
        heading: str | None = None

        if line and not line.startswith("#") and not table_of_contents_pattern.search(line):
            chapter_match = chapter_pattern.match(line)
            article_match = article_pattern.match(line)
            if chapter_match:
                heading = f"## {chapter_match.group(1)}"
            elif article_match:
                heading = f"### {article_match.group(1)}"

        if heading:
            if structured_lines and structured_lines[-1] != "":
                structured_lines.append("")
            structured_lines.extend([heading, ""])
        else:
            structured_lines.append(original_line.rstrip())

    # Collapse excessive blank lines introduced around adjacent headings.
    return re.sub(r"\n{3,}", "\n\n", "\n".join(structured_lines)).strip()


def convert_legal_docs() -> int:
    """Convert PDF/DOCX files trong data/landing/legal/ sang markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    md = MarkItDown()
    manifest = _load_legal_manifest()
    converted = 0

    for filepath in sorted(legal_dir.iterdir()):
        if filepath.suffix.lower() in LEGAL_EXTENSIONS:
            print(f"Converting: {filepath.name}")
            result = md.convert(str(filepath))
            text = result.text_content.strip()
            if not text:
                raise ValueError(f"No text extracted from legal document: {filepath}")
            if filepath.suffix.lower() == ".pdf" and _looks_like_broken_vietnamese(text):
                print("  [INFO] Broken Vietnamese text layer detected; running OCR")
                text = _ocr_pdf(filepath) or text
            text = _structure_legal_markdown(text)

            source = manifest.get(filepath.name, {})
            title = source.get("title", filepath.stem.replace("-", " ").title())
            source_url = source.get("url", "N/A")
            downloaded_at = source.get("downloaded_at", "N/A")
            header = (
                f"# {title}\n\n"
                f"**Source file:** {filepath.name}\n\n"
                f"**Source URL:** {source_url}\n\n"
                f"**Downloaded:** {downloaded_at}\n\n---\n\n"
            )

            output_path = output_dir / f"{filepath.stem}.md"
            output_path.write_text(header + text + "\n", encoding="utf-8")
            converted += 1
            print(f"  [OK] Saved: {output_path}")

    return converted


def convert_news_articles() -> int:
    """Convert JSON crawled articles trong data/landing/news/ sang markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)
    converted = 0

    for filepath in sorted(news_dir.iterdir()):
        if filepath.suffix.lower() == ".json":
            print(f"Converting: {filepath.name}")
            data = json.loads(filepath.read_text(encoding="utf-8"))
            required_fields = ("url", "title", "crawled_at", "content")
            missing = [field for field in required_fields if not data.get(field)]
            if missing:
                raise ValueError(
                    f"{filepath} is missing required field(s): {', '.join(missing)}"
                )

            article_content = data["content"]
            if not isinstance(article_content, str):
                article_content = json.dumps(
                    article_content,
                    ensure_ascii=False,
                    indent=2,
                )
            article_content = article_content.strip()
            if not article_content:
                raise ValueError(f"News article has empty content: {filepath}")

            header = (
                f"# {data['title']}\n\n"
                f"**Source URL:** {data['url']}\n\n"
                f"**Crawled:** {data['crawled_at']}\n\n---\n\n"
            )
            output_path = output_dir / f"{filepath.stem}.md"
            output_path.write_text(
                header + article_content + "\n",
                encoding="utf-8",
            )
            converted += 1
            print(f"  [OK] Saved: {output_path}")

    return converted


def convert_all():
    """Convert toàn bộ files."""
    print("=" * 50)
    print("Task 3: Convert to Markdown (MarkItDown)")
    print("=" * 50)

    print("\n--- Legal Documents ---")
    legal_count = convert_legal_docs()

    print("\n--- News Articles ---")
    news_count = convert_news_articles()

    print(
        f"\n[OK] Converted {legal_count} legal document(s) and "
        f"{news_count} news article(s)."
    )
    print("Output:", OUTPUT_DIR)


if __name__ == "__main__":
    convert_all()
