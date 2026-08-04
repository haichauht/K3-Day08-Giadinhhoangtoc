"""Task 2 - crawl exactly five official UIT articles to JSON."""

from __future__ import annotations

import argparse
import io
import json
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from markitdown import MarkItDown


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "landing" / "news"
ARTICLE_URLS = (
    "https://student.uit.edu.vn/thong-bao-ve-viec-cong-nhan-mon-giao-duc-cam-xuc-xa-hoi-la-mon-tu-chon-tu-do-cho-tat-ca-cac-nganh",
    "https://student.uit.edu.vn/thong-bao-thu-hoc-phi-hoc-ky-2-nam-hoc-2025-2026-trinh-do-dai-hoc",
    "https://daa.uit.edu.vn/content/huong-dan-sinh-vien-dai-hoc-he-chinh-quy-thuc-hien-cac-quy-dinh-ve-chuan-qua-trinh-va-chuan",
    "https://www.uit.edu.vn/bai-viet/giai-ma-cach-xem-thoi-khoa-bieu-cho-tan-sinh-vien-uit",
    "https://ctsv.uit.edu.vn/bai-viet/nhap-hoc-huong-dan-tao-ho-so-truc-tuyen-danh-cho-tan-sinh-vien",
)
USER_AGENT = "Mozilla/5.0 (compatible; UIT-RAG-Lab/1.0)"


def _absolute_links(soup: BeautifulSoup, base_url: str) -> None:
    for element in soup.select("[href]"):
        element["href"] = urljoin(base_url, element["href"])
    for element in soup.select("[src]"):
        element["src"] = urljoin(base_url, element["src"])


def crawl_article(session: requests.Session, url: str) -> dict[str, str]:
    response = session.get(url, timeout=60)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, "html.parser")
    title_node = soup.find("h1") or soup.find("title")
    title = title_node.get_text(" ", strip=True) if title_node else url

    # Retain the complete public page because older UIT sites use inconsistent
    # HTML containers.  Task 3 converts this Markdown and later chunking ranks
    # only relevant sections.
    _absolute_links(soup, url)
    markdown = MarkItDown().convert_stream(
        io.BytesIO(str(soup).encode("utf-8")),
        file_extension=".html",
        url=url,
    ).text_content.strip()
    if not markdown:
        raise ValueError(f"No article content extracted from {url}")
    return {
        "url": url,
        "title": title,
        "crawled_at": datetime.now().astimezone().isoformat(),
        "content": markdown,
    }


def crawl_news(*, force: bool = False) -> list[Path]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    outputs: list[Path] = []

    for index, url in enumerate(ARTICLE_URLS, start=1):
        destination = DATA_DIR / f"article_{index:02d}.json"
        if destination.exists() and destination.stat().st_size > 0 and not force:
            print(f"  Reusing: {destination.name}")
        else:
            payload = crawl_article(session, url)
            destination.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"  Crawled: {destination.name}")
        outputs.append(destination)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Crawl five official UIT articles to JSON")
    parser.add_argument("--force", action="store_true", help="Crawl and overwrite existing JSON")
    args = parser.parse_args()
    outputs = crawl_news(force=args.force)
    print(f"Ready: {len(outputs)} news JSON file(s) in {DATA_DIR}")


if __name__ == "__main__":
    main()
