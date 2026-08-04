"""Task 1 - collect three official UIT/ĐHQG-HCM legal PDFs."""

from __future__ import annotations

import argparse
from pathlib import Path

import requests


DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "landing" / "legal"
LEGAL_DOCUMENTS = (
    (
        "https://daa.uit.edu.vn/sites/daa/files/202412/507-qd-dhcntt-27-5-2024_quy_che_dao_tao_cho_sinh_vien.pdf",
        "quy-che-dao-tao-tu-xa-uit.pdf",
    ),
    (
        "https://daa.uit.edu.vn/sites/daa/files/202510/qd2062_221025_quy_dinh_khoi_hp_cong_nhanchuyen_doi_tin_chi.signed.pdf",
        "quy-dinh-cong-nhan-chuyen-doi-tin-chi-dhqg-hcm.pdf",
    ),
    (
        "https://daa.uit.edu.vn/sites/daa/files/202310/1195-qd-dhqg_27-9-2019_quy_dinh_dao_tao_song_nganh_dhqg.pdf",
        "quy-dinh-dao-tao-song-nganh-dhqg-hcm.pdf",
    ),
)


def setup_directory() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


def download_file(url: str, filename: str, *, force: bool = False) -> Path:
    destination = setup_directory() / filename
    if destination.exists() and destination.stat().st_size > 0 and not force:
        print(f"  Reusing: {destination.name}")
        return destination

    response = requests.get(
        url,
        timeout=60,
        headers={"User-Agent": "UIT-RAG-Lab/1.0"},
    )
    response.raise_for_status()
    payload = response.content
    if len(payload) < 1024 or not payload.startswith(b"%PDF"):
        raise ValueError(f"Source did not return a valid PDF: {url}")

    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(destination)
    print(f"  Downloaded: {destination.name} ({len(payload):,} bytes)")
    return destination


def collect_legal_documents(*, force: bool = False) -> list[Path]:
    return [download_file(url, filename, force=force) for url, filename in LEGAL_DOCUMENTS]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download three official UIT legal PDFs")
    parser.add_argument("--force", action="store_true", help="Download again even if files exist")
    args = parser.parse_args()
    files = collect_legal_documents(force=args.force)
    print(f"Ready: {len(files)} legal PDF(s) in {DATA_DIR}")


if __name__ == "__main__":
    main()
