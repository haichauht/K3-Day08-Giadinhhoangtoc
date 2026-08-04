"""Crawl news from https://www.uit.edu.vn/tin-uit to Markdown files.

Examples:
    python src/task2_crawl_news.py
    python src/task2_crawl_news.py --pages 2 --delay 1
"""

from __future__ import annotations

import argparse
import io
import re
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup, Tag
from markitdown import MarkItDown

LIST_URL = "https://www.uit.edu.vn/tin-uit"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "landing" / "news"
USER_AGENT = (
    "Mozilla/5.0 (compatible; UITNewsCrawler/1.0; "
    "+https://www.uit.edu.vn/tin-uit)"
)


def yaml_string(value: str) -> str:
    """Return a double-quoted YAML scalar without adding a YAML dependency."""
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    escaped = escaped.replace("\r", " ").replace("\n", " ")
    return f'"{escaped}"'


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


def html_to_markdown(fragment: Tag, source_url: str) -> str:
    """Convert the selected article HTML with the locally installed MarkItDown."""
    fragment = deepcopy(fragment)
    make_links_absolute(fragment, source_url)
    html = str(fragment).encode("utf-8")
    result = MarkItDown().convert_stream(
        io.BytesIO(html), file_extension=".html", url=source_url
    )
    return result.text_content.strip()


def crawl_article(session: requests.Session, url: str) -> tuple[str, str]:
    """Return (filename, complete Markdown document) for one UIT article."""
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

    content_markdown = html_to_markdown(content, url)
    slug = slug_from_url(url)
    retrieved_at = datetime.now().astimezone().date().isoformat()

    front_matter = [
        "---",
        f"doc_id: {yaml_string(f'uit-news-{slug}')}",
        f"title: {yaml_string(title)}",
        f"source_url: {yaml_string(url)}",
        'source_section: "Tin UIT"',
        f"retrieved_at: {yaml_string(retrieved_at)}",
        'document_version: "not-stated"',
        f"page_published_at: {yaml_string(published_at or 'not-stated')}",
        'audience: "public"',
        'institution: "uit"',
        'department: "communications"',
        'category: "news"',
        'language: "vi"',
        'cleaning_version: "markitdown-v1"',
        "---",
    ]
    document = "\n".join(front_matter) + f"\n\n# {title}\n\n{content_markdown}\n"
    return f"{slug}.md", document


def crawl(pages: int, output_dir: Path, delay: float) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    article_urls = collect_article_urls(session, pages)
    if not article_urls:
        raise RuntimeError("No UIT article URLs were found")

    saved = 0
    for index, article_url in enumerate(article_urls, start=1):
        print(f"[{index}/{len(article_urls)}] Crawling: {article_url}")
        try:
            filename, document = crawl_article(session, article_url)
            destination = output_dir / filename
            destination.write_text(document, encoding="utf-8")
            saved += 1
            print(f"  Saved: {destination}")
        except (requests.RequestException, ValueError) as error:
            print(f"  Skipped: {error}")

        if delay > 0 and index < len(article_urls):
            time.sleep(delay)

    return saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crawl UIT news to Markdown")
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
    args = parser.parse_args()
    if args.pages < 1:
        parser.error("--pages must be at least 1")
    if args.delay < 0:
        parser.error("--delay cannot be negative")
    return args


def main() -> None:
    args = parse_args()
    saved = crawl(args.pages, args.output, args.delay)
    print(f"Done: saved {saved} Markdown file(s) to {args.output.resolve()}")


if __name__ == "__main__":
    main()
