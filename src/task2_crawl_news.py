"""Crawl UIT news into raw JSON snapshots in the landing layer.

Examples:
    python src/task2_crawl_news.py
    python src/task2_crawl_news.py --pages 10 --limit 5 --delay 1
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup, Tag

LIST_URL = "https://www.uit.edu.vn/tin-uit"
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "landing" / "news"
SOURCE_CSV_FIELDS = (
    "doc_id",
    "file_path",
    "title",
    "source_url",
    "retrieved_at",
    "document_version",
    "license_or_permission",
    "cleaning_version",
)
USER_AGENT = (
    "Mozilla/5.0 (compatible; UITNewsCrawler/1.0; +https://www.uit.edu.vn/tin-uit)"
)


def slug_from_url(url: str) -> str:
    slug = Path(urlparse(url).path).name.strip().lower()
    slug = re.sub(r"[^a-z0-9-]+", "-", slug).strip("-")
    return slug or "article"


def listing_url(page: int) -> str:
    return LIST_URL if page == 1 else f"{LIST_URL}?page={page}"


def collect_article_urls(session: requests.Session, pages: int) -> list[str]:
    """Collect unique article URLs from the requested listing pages."""
    urls: list[str] = []
    seen: set[str] = set()

    for page in range(1, pages + 1):
        url = listing_url(page)
        response = session.get(url, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # News cards use an h3 title. Restricting the selector avoids static
        # /bai-viet/ links that also appear in the site's navigation menus.
        page_urls = [
            urljoin(url, link["href"])
            for link in soup.select('h3 a[href^="/bai-viet/"]')
        ]
        for article_url in page_urls:
            if article_url not in seen:
                seen.add(article_url)
                urls.append(article_url)

        print(f"Listing page {page}: found {len(set(page_urls))} article(s)")

    return urls


def largest_article(soup: BeautifulSoup) -> Tag:
    articles = soup.find_all("article")
    if not articles:
        raise ValueError("Could not find an <article> element")
    return max(articles, key=lambda item: len(item.get_text(" ", strip=True)))


def parse_published_at(article: Tag) -> tuple[str, str]:
    """Return (ISO date, original text); empty strings mean not found."""
    date_match = re.search(
        r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})(?:\s*-\s*\d{1,2}:\d{2})?\b",
        article.get_text(" ", strip=True),
    )
    if not date_match:
        return "", ""

    original = date_match.group(0)
    day, month, year = map(int, date_match.groups())
    try:
        iso_date = datetime(year, month, day).date().isoformat()
    except ValueError:
        return "", original
    return iso_date, original


def make_links_absolute(fragment: Tag, base_url: str) -> None:
    for tag in fragment.select("[href]"):
        tag["href"] = urljoin(base_url, tag["href"])
    for tag in fragment.select("[src]"):
        tag["src"] = urljoin(base_url, tag["src"])


def crawl_article(session: requests.Session, url: str) -> tuple[str, dict[str, str]]:
    """Return a filename and raw, self-contained JSON-serializable article."""
    response = session.get(url, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    article = largest_article(soup)

    title_tag = article.find("h1") or soup.find("h1")
    if title_tag is None:
        raise ValueError("Could not find the article title")
    title = title_tag.get_text(" ", strip=True)

    published_at, published_text = parse_published_at(article)
    content = article.select_one(".rich-text-content")
    if content is None:
        # Fallback for older pages with a slightly different template.
        content = deepcopy(article)
        for heading in content.find_all("h1"):
            heading.decompose()
        if published_text:
            for node in content.find_all(string=re.compile(re.escape(published_text))):
                if node.parent:
                    node.parent.decompose()

    slug = slug_from_url(url)
    retrieved_at = datetime.now().astimezone().date().isoformat()
    content_copy = deepcopy(content)
    make_links_absolute(content_copy, url)
    payload = {
        "url": url,
        "title": title,
        "published_at": published_at or "not-stated",
        "retrieved_at": retrieved_at,
        "content_html": str(content_copy),
        "content_text": content_copy.get_text(" ", strip=True),
    }
    return f"{slug}.json", payload


def crawl(pages: int, limit: int, output_dir: Path, delay: float) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    article_urls = collect_article_urls(session, pages)[:limit]
    if not article_urls:
        raise RuntimeError("No UIT article URLs were found")

    saved = 0
    for index, article_url in enumerate(article_urls, start=1):
        print(f"[{index}/{len(article_urls)}] Crawling: {article_url}")
        try:
            filename, payload = crawl_article(session, article_url)
            destination = output_dir / filename
            destination.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            saved += 1
            print(f"  Saved: {destination}")
        except (requests.RequestException, ValueError) as error:
            print(f"  Skipped: {error}")

        if delay > 0 and index < len(article_urls):
            time.sleep(delay)

    return saved


def read_front_matter(markdown_path: Path) -> dict[str, str]:
    """Read scalar metadata from a generated standardized Markdown file."""
    if not markdown_path.exists():
        return {}
    lines = markdown_path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != "---":
        return {}
    metadata: dict[str, str] = {}
    for line in lines[1:]:
        if line == "---":
            break
        key, separator, value = line.partition(":")
        if separator:
            value = value.strip()
            try:
                metadata[key.strip()] = str(json.loads(value))
            except json.JSONDecodeError:
                metadata[key.strip()] = value.strip("'\"")
    return metadata


def rebuild_source_csv(output_dir: Path) -> tuple[Path, int]:
    """Rebuild landing/source.csv, pointing at standardized Markdown outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []

    for json_path in sorted(output_dir.glob("*.json")):
        try:
            metadata = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as error:
            print(f"  Source index skipped ({json_path.name}): {error}")
            continue
        required = ("url", "title", "retrieved_at")
        if any(not metadata.get(field) for field in required):
            print(f"  Source index skipped (missing metadata): {json_path.name}")
            continue
        slug = slug_from_url(metadata["url"])
        relative_markdown = Path("data") / "standardized" / "news" / f"{slug}.md"
        standardized = read_front_matter(REPO_ROOT / relative_markdown)
        rows.append(
            {
                "doc_id": standardized.get("doc_id", f"uit-news-{slug}"),
                "file_path": relative_markdown.as_posix(),
                "title": standardized.get("title", metadata["title"]),
                "source_url": standardized.get("source_url", metadata["url"]),
                "retrieved_at": standardized.get(
                    "retrieved_at", metadata["retrieved_at"]
                ),
                "document_version": standardized.get("document_version", "not-stated"),
                "license_or_permission": "public-page",
                "cleaning_version": standardized.get(
                    "cleaning_version",
                    "markitdown-html-v2"
                    if metadata.get("content_html")
                    else "markitdown-v1",
                ),
            }
        )

    rows.sort(key=lambda row: row["file_path"])
    source_path = output_dir / "source.csv"
    with source_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=SOURCE_CSV_FIELDS,
            quoting=csv.QUOTE_ALL,
        )
        writer.writeheader()
        writer.writerows(rows)
    return source_path, len(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crawl UIT news to landing JSON")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of newest articles to retain (default: 5)",
    )
    parser.add_argument(
        "--pages",
        type=int,
        default=1,
        help="Number of listing pages to crawl (default: 1)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA_DIR,
        help=f"Output directory (default: {DATA_DIR})",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay in seconds between article requests (default: 0.5)",
    )
    parser.add_argument(
        "--rebuild-source-csv",
        action="store_true",
        help="Only rebuild source.csv from existing JSON snapshots",
    )
    args = parser.parse_args()
    if args.pages < 1:
        parser.error("--pages must be at least 1")
    if args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.delay < 0:
        parser.error("--delay cannot be negative")
    return args


def main() -> None:
    args = parse_args()
    if args.rebuild_source_csv:
        source_path, row_count = rebuild_source_csv(args.output)
        print(f"Done: indexed {row_count} JSON snapshot(s) in {source_path.resolve()}")
        return

    saved = crawl(args.pages, args.limit, args.output, args.delay)
    source_path, row_count = rebuild_source_csv(args.output)
    print(f"Done: saved {saved} JSON snapshot(s) to {args.output.resolve()}")
    print(f"Source index: {row_count} row(s) in {source_path.resolve()}")


if __name__ == "__main__":
    main()
