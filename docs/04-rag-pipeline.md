# 04 - Pipeline RAG (Retrieval-Augmented Generation)

## Tổng quan

Pipeline RAG kết hợp 3 kỹ thuật retrieval + graph expansion + LLM generation:

```
Query → Classification → Hybrid Retrieval → (Rerank) → Graph Expansion → LLM → Answer
```

## 1. Classification (Lọc câu hỏi)

**Module**: `src/domain/classification/classify.py`

| Kết quả | Ý nghĩa | Hành động |
|---------|---------|-----------|
| `-1` | Không phải tiếng Việt | Trả lời: "Tôi chỉ hiểu tiếng Việt..." |
| `0` | Không liên quan giao thông | Trả lời: "Câu hỏi không liên quan..." |
| `1` | Hợp lệ | Tiếp tục RAG pipeline |

**Logic**:
1. Detect language (langdetect) → reject nếu != `vi`
2. Preprocess query (lowercase + VnCoreNLP word segment + remove stopwords)
3. Keyword matching: nếu query chứa ít nhất 1 keyword trong `top_keywords.txt` → pass

## 2. Hybrid Retrieval

**Module**: `src/domain/Retrieval/retrieval.py`

### Vector Search (weight = 0.6)
- Embedding model: `amazon.titan-embed-text-v2:0` (1024 dimensions)
- Vector store: ChromaDB
- Top-K: 10 (configurable)
- Hỗ trợ metadata filtering (Chroma `$and` where-filter)

### BM25 Keyword Search (weight = 0.4)
- LangChain `BM25Retriever`
- Hoạt động trên toàn bộ corpus documents
- Cùng Top-K = 10

### Ensemble
- `EnsembleRetriever` kết hợp 2 retriever
- Weights configurable: `BM25_WEIGHT=0.4`, `VECTOR_WEIGHT=0.6`
- Enable/disable qua `ENABLE_HYBRID=true/false`

## 3. Rerank (Optional)

- Model: FlashrankRerank
- Top-N sau rerank: 4 (configurable)
- Enable qua `ENABLE_RERANK=true` + `pip install flashrank`
- Wrap bằng `ContextualCompressionRetriever`

## 4. Graph Expansion (LegalGraph)

**Module**: `src/domain/Retrieval/legal_graph.py`

### Cấu trúc graph
- **Nodes**: doc_id, chương, điều, khoản
- **Edges**:
  - Hierarchical: doc ↔ chương ↔ điều ↔ khoản
  - Cross-reference: điều A → điều B (từ field `references`)

### Expansion algorithm
- BFS từ seed documents (kết quả retrieval)
- Max hops: 1
- Max expand: 5 documents bổ sung
- Mục đích: bổ sung context liên quan (VD: câu hỏi về Điều 5 → kéo thêm Điều 12 được tham chiếu)

## 5. LLM Generation

### Model
- `anthropic.claude-3-haiku-20240307-v1:0` (AWS Bedrock)
- Region: `us-east-1`

### Prompt Template
```
Bạn là trợ lý ảo giúp trả lời các câu hỏi về luật giao thông đường bộ.
Chỉ trả lời dựa trên ngữ cảnh và lịch sử hội thoại được cung cấp, không thêm thông tin bên ngoài.

Ngữ cảnh:
---------------------
{context}
---------------------
Lịch sử hội thoại:
{history}

Câu hỏi: {question}
Câu trả lời (kèm trích dẫn từ tiêu đề):
```

### Context formatting
Mỗi document được format:
```
Tiêu đề: [metadata.title]
Nội dung: [page_content]
```

## 6. Metadata Filtering

Hỗ trợ filter theo metadata khi query:
```python
retrieval.query(question, metadata_filter={"doc_id": "36/2024/QH15", "chuong": "III"})
```

Chroma filter được build tự động:
- Single value: `{"key": {"$eq": value}}`
- List value: `{"key": {"$in": [values]}}`
- Multiple keys: `{"$and": [...]}`

## 7. Cấu hình Pipeline

| Biến môi trường | Mặc định | Mô tả |
|----------------|----------|-------|
| `TOP_K_RESULTS` | 10 | Số documents trả về |
| `ENABLE_HYBRID` | true | Bật hybrid search |
| `ENABLE_RERANK` | false | Bật reranking |
| `RERANK_TOP_N` | 4 | Số docs sau rerank |
| `BM25_WEIGHT` | 0.4 | Trọng số BM25 |
| `VECTOR_WEIGHT` | 0.6 | Trọng số vector |
| `CHROMA_COLLECTION` | AI002 | Tên collection ChromaDB |
| `EMBEDDING_MODEL_ID` | amazon.titan-embed-text-v2:0 | Model embedding |
| `LLM_MODEL_ID` | anthropic.claude-3-haiku-20240307-v1:0 | Model LLM |
