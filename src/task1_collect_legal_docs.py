"""Task 1 - Validate collected legal documents and their provenance metadata."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "landing" / "legal"
SOURCES_PATH = DATA_DIR / "sources.json"
MINIMUM_DOCUMENTS = 3


def setup_directory() -> Path:
    """Create and return the legal landing directory."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


def validate_legal_landing() -> dict:
    """Verify that every source record points to a collected local document."""
    setup_directory()
    if not SOURCES_PATH.exists():
        raise FileNotFoundError(f"Missing provenance manifest: {SOURCES_PATH}")

    sources = json.loads(SOURCES_PATH.read_text(encoding="utf-8"))
    if not isinstance(sources, list) or len(sources) < MINIMUM_DOCUMENTS:
        raise ValueError(
            f"Expected at least {MINIMUM_DOCUMENTS} legal sources, got {len(sources)}"
        )

    required = {"title", "url", "filename", "downloaded_at"}
    for index, source in enumerate(sources, start=1):
        missing = required - source.keys()
        if missing:
            raise ValueError(f"Source {index} is missing: {sorted(missing)}")
        document_path = DATA_DIR / source["filename"]
        if not document_path.is_file():
            raise FileNotFoundError(f"Missing collected document: {document_path}")

    return {
        "documents": len(sources),
        "manifest": SOURCES_PATH.relative_to(REPO_ROOT).as_posix(),
        "status": "valid",
    }


if __name__ == "__main__":
    print(json.dumps(validate_legal_landing(), ensure_ascii=False, indent=2))
