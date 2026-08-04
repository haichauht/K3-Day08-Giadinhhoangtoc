"""
Task 1 — Thu thập văn bản chính sách/quy định dịch vụ đại học.

Hướng dẫn:
    1. Tìm tối thiểu 3 văn bản chính sách (PDF/DOCX) từ trang công khai của một trường đại học.
    2. Tải về và lưu vào data/landing/legal/
    3. Đặt tên file rõ ràng, không dấu, mô tả đúng nội dung.

Gợi ý nguồn (ví dụ trang công khai RMIT Vietnam — rmit.edu.vn):
    - https://www.rmit.edu.vn/study-at-rmit/tuition-fees
    - https://www.rmit.edu.vn/study-at-rmit/scholarships/...
    - https://www.rmit.edu.vn/students/my-studies/fees-and-payments

Gợi ý văn bản (chủ đề dịch vụ đại học):
    - Học phí & phương thức thanh toán (Tuition Fees)
    - Chính sách học bổng (Scholarship eligibility)
    - Quy định ký túc xá / hỗ trợ chỗ ở (Accommodation Services)
    - Hướng dẫn đăng ký học phần qua cổng thông tin sinh viên (Course Registration)

Lưu ý: một số trang trường (vd VinUni, Fulbright) chặn bot crawler mặc định (HTTP 403) —
không phải lỗi của bạn, đó là cấu hình WAF/Cloudflare phía server. Đổi sang trang khác
thay vì cố vượt qua, và chỉ dùng nguồn công khai/được phép chia sẻ.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import requests

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"
SOURCES_PATH = DATA_DIR / "sources.json"

# Văn bản pháp lý/quy định học vụ công khai của UIT / ĐHQG-HCM (daa.uit.edu.vn).
LEGAL_DOCS = [
    {
        "title": "Quy định đào tạo song ngành trình độ đại học của ĐHQG-HCM",
        "url": "https://daa.uit.edu.vn/sites/daa/files/202310/1195-qd-dhqg_27-9-2019_quy_dinh_dao_tao_song_nganh_dhqg.pdf",
        "filename": "quy-dinh-dao-tao-song-nganh-dhqg-hcm.pdf",
    },
    {
        "title": "Quy chế đào tạo từ xa trình độ đại học của UIT",
        "url": "https://daa.uit.edu.vn/sites/daa/files/202412/507-qd-dhcntt-27-5-2024_quy_che_dao_tao_cho_sinh_vien.pdf",
        "filename": "quy-che-dao-tao-tu-xa-uit.pdf",
    },
    {
        "title": "Quy định công nhận và chuyển đổi tín chỉ tại ĐHQG-HCM",
        "url": "https://daa.uit.edu.vn/sites/daa/files/202510/qd2062_221025_quy_dinh_khoi_hp_cong_nhanchuyen_doi_tin_chi.signed.pdf",
        "filename": "quy-dinh-cong-nhan-chuyen-doi-tin-chi-dhqg-hcm.pdf",
    },
]

USER_AGENT = "Mozilla/5.0 (compatible; UITLegalDocsCollector/1.0)"


def setup_directory():
    """Tạo thư mục data/landing/legal/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Thư mục đã sẵn sàng: {DATA_DIR}")


def download_file(url: str, filename: str) -> Path:
    """Tải 1 file PDF về DATA_DIR, bỏ qua nếu đã tồn tại."""
    filepath = DATA_DIR / filename
    if filepath.exists():
        print(f"  = Đã có sẵn, bỏ qua: {filepath.name}")
        return filepath

    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    response.raise_for_status()
    filepath.write_bytes(response.content)
    print(f"  Đã tải: {filepath.name} ({len(response.content):,} bytes)")
    return filepath


def collect_legal_docs() -> list[dict]:
    """Tải toàn bộ LEGAL_DOCS về data/landing/legal/ và ghi provenance vào sources.json."""
    setup_directory()

    sources = []
    for doc in LEGAL_DOCS:
        filepath = download_file(doc["url"], doc["filename"])
        sources.append({
            "title": doc["title"],
            "url": doc["url"],
            "filename": doc["filename"],
            "downloaded_at": datetime.now(timezone.utc).isoformat(),
            "size_bytes": filepath.stat().st_size,
        })

    SOURCES_PATH.write_text(json.dumps(sources, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✓ Saved provenance: {SOURCES_PATH}")
    return sources


if __name__ == "__main__":
    collect_legal_docs()
