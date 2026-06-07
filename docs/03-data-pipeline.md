# 03 - Pipeline xử lý dữ liệu

## Tổng quan

Pipeline chuyển đổi file PDF văn bản pháp luật thành các records có cấu trúc (JSON), sẵn sàng để embedding và lưu vào vector store.

```
PDF files → Text Extraction → Structure Parsing → Chunking → JSON records → ChromaDB
```

## Nguồn dữ liệu

| File | Mã văn bản | Loại | Ghi chú |
|------|-----------|------|---------|
| `36-2024-qh15.pdf` | 36/2024/QH15 | Luật | Luật TTATGT đường bộ (phần 1) |
| `36-2024-qh15_tiep.pdf` | 36/2024/QH15 | Luật | Luật TTATGT đường bộ (phần 2) |
| `168-nd-cp.signed.pdf` | 168/2024/NĐ-CP | Nghị định | NĐ xử phạt vi phạm HC |

## Quy trình chi tiết

### Bước 1: Text Extraction (`crawl_data.py`)

**Ưu tiên text layer** (PyMuPDF `fitz`):
- Đọc text trực tiếp từ PDF
- Nếu mật độ text quá thấp (< 50 ký tự/trang) → fallback sang OCR

**OCR fallback** (Tesseract + PIL):
- Render page thành image (DPI=300)
- OCR bằng Tesseract với lang=`vie`
- Cần `tessdata` cho tiếng Việt

### Bước 2: Noise Removal

Loại bỏ các dòng nhiễu bằng regex:
- Chữ ký số: `Người ký:`, `Email:`, `Cơ quan:`, `Thời gian ký:`
- Header công báo: `CÔNG BÁO/Số`
- Số trang đơn lẻ: `^\d{1,4}$`

### Bước 3: Structure-aware Parsing

Nhận diện cấu trúc pháp luật bằng regex:
```
Chương [I, II, ...] → chuong
Mục [1, 2, ...]     → muc
Điều [số]. [tiêu đề] → dieu, dieu_title
[số]. [nội dung]     → khoan (khoản)
```

### Bước 4: Chunking

**Đơn vị chunk tối thiểu = Khoản** (không bao giờ cắt giữa khoản).

Mỗi record chứa:
```json
{
  "title": "36/2024/QH15 - Điều 5. Nguyên tắc bảo đảm TTATGT - Khoản 2",
  "content": "Điều 5. Nguyên tắc bảo đảm TTATGT\n2. Mọi hành vi vi phạm...",
  "doc_id": "36/2024/QH15",
  "doc_type": "luat",
  "chuong": "I",
  "muc": "",
  "dieu": 5,
  "dieu_title": "Nguyên tắc bảo đảm trật tự, an toàn giao thông đường bộ",
  "khoan": 2,
  "references": ["Điều 12", "Điều 35"]
}
```

### Bước 5: Cross-references

Tự động trích xuất tham chiếu chéo (`Điều X`) trong nội dung mỗi khoản → field `references`.

## Output

File: `data/output.json`
- Mảng JSON các records
- Mỗi record = 1 khoản (hoặc 1 điều nếu không có khoản)
- Metadata đầy đủ cho filtering và graph expansion

## Chạy pipeline

```bash
python src/data_preparation/crawl_data.py
```

Output: `data/output.json`

## Text Preprocessing (cho query)

Module `src/utils/text_preprocessing.py`:

1. **Language detection** (`langdetect`) → reject nếu không phải tiếng Việt
2. **Lowercasing**
3. **Word segmentation** (VnCoreNLP) → tách từ ghép tiếng Việt
4. **Stopwords removal** → loại bỏ từ dừng từ file `vietnamese-stopwords-dash.txt`

## Keyword Extraction

Module `src/domain/classification/extract_keyword.py`:
- Trích xuất top keywords từ corpus luật giao thông
- Output: `data/top_keywords.txt` (format: `keyword: count`)
- Dùng cho RuleBasedClassifier để lọc câu hỏi off-topic
