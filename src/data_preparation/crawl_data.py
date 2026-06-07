# crawl_data.py — extract structured, metadata-rich records from traffic-law PDFs.
# Chunking is structure-aware: atomic unit = Khoản (never split mid-clause).
from __future__ import annotations

import json
import os
import re
import sys

import fitz  # PyMuPDF

try:  # Windows consoles default to cp1252; force UTF-8 so Vietnamese logs don't crash.
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# OCR config (override via env for Docker/Linux). On Windows the UB-Mannheim default path is used.
_TESSERACT_CMD = os.getenv("TESSERACT_CMD") or r"C:\Program Files\Tesseract-OCR\tesseract.exe"
_DEF_TESSDATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "tessdata"))
_TESSDATA_DIR = os.getenv("TESSDATA_PREFIX") or (_DEF_TESSDATA if os.path.isdir(_DEF_TESSDATA) else None)

# Boilerplate / noise (digital signature block, gazette header, standalone page numbers).
_NOISE = [
    re.compile(r"^Người ký:", re.I),
    re.compile(r"^Email:", re.I),
    re.compile(r"^Cơ quan:", re.I),
    re.compile(r"^Thời gian ký:", re.I),
    re.compile(r"CÔNG BÁO/Số", re.I),
    re.compile(r"^\d{1,4}$"),
]
_CHUONG = re.compile(r"^Chương\s+([IVXLCDM\d]+)\b", re.I)
_MUC = re.compile(r"^Mục\s+([IVXLCDM\d]+)\b", re.I)
_DIEU = re.compile(r"^Điều\s+(\d+)\.\s*(.*)", re.I)
_KHOAN = re.compile(r"^(\d+)\.\s")


def _is_noise(line: str) -> bool:
    return any(p.search(line) for p in _NOISE)


def _refs(text: str) -> list[str]:
    return sorted(set(re.findall(r"Điều\s+\d+", text)))


def extract_text(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    text = "".join(page.get_text("text") for page in doc)
    doc.close()
    return text


def _configure_tesseract():
    import pytesseract

    if os.path.exists(_TESSERACT_CMD):
        pytesseract.pytesseract.tesseract_cmd = _TESSERACT_CMD
    return pytesseract


def _ocr_available() -> bool:
    try:
        _configure_tesseract().get_tesseract_version()
        return True
    except Exception:
        return False


def extract_text_ocr(pdf_path: str, lang: str = "vie", dpi: int = 300) -> str:
    import io

    from PIL import Image

    pytesseract = _configure_tesseract()
    cfg = f"--tessdata-dir {_TESSDATA_DIR}" if _TESSDATA_DIR else ""
    doc = fitz.open(pdf_path)
    parts = []
    for page in doc:
        img = Image.open(io.BytesIO(page.get_pixmap(dpi=dpi).tobytes("png")))
        parts.append(pytesseract.image_to_string(img, lang=lang, config=cfg))
    doc.close()
    return "\n".join(parts)


def load_pdf_text(pdf_path: str, ocr_lang: str = "vie") -> str:
    """Use the text layer; fall back to OCR when the PDF is scanned (very low text density)."""
    doc = fitz.open(pdf_path)
    pages = doc.page_count
    doc.close()
    text = extract_text(pdf_path)
    if len(text.strip()) >= 50 * max(pages, 1):
        return text
    if not _ocr_available():
        print(f"  [Cần OCR nhưng Tesseract chưa sẵn sàng -> bỏ qua] {pdf_path}")
        return text
    print(f"  PDF scan: đang OCR {pages} trang (lang={ocr_lang}, có thể mất vài phút)...")
    return extract_text_ocr(pdf_path, lang=ocr_lang)


def extract_records(text: str, doc_id: str, doc_type: str) -> list[dict]:
    chuong = muc = ""
    dieu_no = None
    dieu_title = ""
    khoan_no = None
    buf: list[str] = []
    records: list[dict] = []

    def flush():
        nonlocal buf
        if dieu_no is None or not " ".join(buf).strip():
            buf = []
            return
        body = " ".join(buf).strip()
        header = f"Điều {dieu_no}. {dieu_title}".strip()
        records.append({
            "title": f"{doc_id} - {header}" + (f" - Khoản {khoan_no}" if khoan_no else ""),
            "content": f"{header}\n{body}",
            "doc_id": doc_id,
            "doc_type": doc_type,
            "chuong": chuong,
            "muc": muc,
            "dieu": dieu_no,
            "dieu_title": dieu_title,
            "khoan": khoan_no or 0,
            "references": _refs(body),
        })
        buf = []

    for raw in text.splitlines():
        line = raw.strip()
        if not line or _is_noise(line):
            continue

        if (m := _CHUONG.match(line)):
            flush(); chuong = m.group(1); muc = ""; continue
        if (m := _MUC.match(line)):
            flush(); muc = m.group(1); continue
        if (m := _DIEU.match(line)):
            flush(); dieu_no = int(m.group(1)); dieu_title = m.group(2).strip(); khoan_no = None; continue
        if (m := _KHOAN.match(line)) and dieu_no is not None:
            flush(); khoan_no = int(m.group(1)); buf = [line]; continue

        buf.append(line)

    flush()
    return records


if __name__ == "__main__":
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    data_dir = os.path.join(root, "data")
    sources = [
        ("36-2024-qh15.pdf", "36/2024/QH15", "luat"),
        ("36-2024-qh15_tiep.pdf", "36/2024/QH15", "luat"),
        ("168-nd-cp.signed.pdf", "168/2024/NĐ-CP", "nghi_dinh"),
    ]

    all_records: list[dict] = []
    for fname, doc_id, doc_type in sources:
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            print(f"Bỏ qua (không tìm thấy): {path}")
            continue
        print(f"Trích xuất: {fname}")
        all_records.extend(extract_records(load_pdf_text(path), doc_id, doc_type))

    out = os.path.join(data_dir, "output.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)
    print(f"Đã lưu {len(all_records)} records -> {out}")
