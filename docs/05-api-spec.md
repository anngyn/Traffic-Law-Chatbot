# 05 - API Specification

## Base URL
```
http://localhost:8000
```

## Authentication

- Header: `X-Api-Key`
- Auth **disabled** khi biến `API_KEYS` rỗng (mặc định dev mode)
- Khi enabled: request thiếu/sai key → `401 Unauthorized`

## Rate Limiting

- Mặc định: `30/minute` per IP
- Configurable qua `RATE_LIMIT` env var
- Vượt limit → `429 Too Many Requests`

---

## Endpoints

### GET /health

Health check endpoint.

**Response** `200 OK`:
```json
{
  "status": "ok",
  "model": "anthropic.claude-3-haiku-20240307-v1:0"
}
```

---

### POST /chat

Gửi câu hỏi và nhận câu trả lời từ chatbot.

**Headers**:
```
Content-Type: application/json
X-Api-Key: <key>  (nếu auth enabled)
```

**Request Body**:
```json
{
  "message": "string (required) - Câu hỏi của user",
  "session_id": "string | null (optional) - ID phiên hội thoại"
}
```

**Response** `200 OK`:
```json
{
  "answer": "string - Câu trả lời từ chatbot"
}
```

**Error Responses**:
| Status | Mô tả |
|--------|--------|
| 401 | Invalid or missing API key |
| 429 | Rate limit exceeded |
| 422 | Validation error (thiếu field `message`) |
| 500 | Internal server error |

---

## Luồng xử lý /chat

1. Validate request body (Pydantic)
2. Check API key (nếu enabled)
3. Check rate limit
4. Load chat history từ DB (nếu có `session_id`)
5. Check cache (SHA1 hash của session_id + history + message)
6. Nếu cache miss → gọi `ChatBot.process_query(message, history)`
7. Lưu response vào cache
8. Lưu message vào chat history (nếu có `session_id`)
9. Return answer

## Ví dụ sử dụng

### cURL
```bash
# Health check
curl http://localhost:8000/health

# Chat (không auth)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Mức phạt vượt đèn đỏ là bao nhiêu?", "session_id": "user123"}'

# Chat (có auth)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-Api-Key: my-secret-key" \
  -d '{"message": "Điều 5 luật giao thông quy định gì?"}'
```

### Python
```python
import requests

resp = requests.post("http://localhost:8000/chat", json={
    "message": "Xe máy được chạy tối đa bao nhiêu km/h trong khu dân cư?",
    "session_id": "session-abc"
})
print(resp.json()["answer"])
```

## Data Models

### ChatRequest
```python
class ChatRequest(BaseModel):
    message: str            # Câu hỏi (bắt buộc)
    session_id: str | None  # Session ID cho multi-turn (tùy chọn)
```

### ChatResponse
```python
class ChatResponse(BaseModel):
    answer: str  # Câu trả lời từ chatbot
```
