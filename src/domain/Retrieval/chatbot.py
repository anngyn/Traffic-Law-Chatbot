# chatbot.py
from __future__ import annotations

import os
import sys
from typing import Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Retrieval.database import ChromaVectorStoreManager
from Retrieval.retrieval import Retrieval
from classification.classify import RuleBasedClassifier

# Keywords that signal the user wants penalty info → route to NĐ-CP corpus only
_PENALTY_KEYWORDS = [
    "phạt", "bị phạt", "tiền phạt", "xử phạt", "mức phạt",
    "tước", "tước bằng", "tước giấy phép",
    "trừ điểm", "bao nhiêu tiền", "phạt bao nhiêu",
    "vi phạm bị", "bị xử", "nghị định 168",
]
_PENALTY_DOC_FILTER = {"doc_id": "168/2024/NĐ-CP"}


def _detect_penalty_intent(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in _PENALTY_KEYWORDS)


class ChatBot:
    """Điều phối pipeline RAG luật giao thông (LangChain hybrid + graph expansion + AWS Bedrock)."""

    def __init__(self, stopwords_path: str, folder_path: str, keyword_file: str, processed_json_file: str):
        self.database = ChromaVectorStoreManager(data_folder=folder_path)
        self.classifier = RuleBasedClassifier(keyword_file=keyword_file, stopwords_path=stopwords_path)

        documents = self.database.load_documents(processed_json_file)
        if self.database.count_nodes() == 0:
            print("Không tìm thấy dữ liệu. Đang tạo chỉ mục mới từ tài liệu...")
            self.database.store(documents)

        self.retrieval = Retrieval(vectorstore=self.database.vectorstore, documents=documents)

    def process_query(self, user_question: str, history: str = "", metadata_filter: dict[str, Any] | None = None) -> str:
        result_class = self.classifier(user_question)
        if result_class == -1:
            return "Tôi chỉ hiểu tiếng Việt. Bạn vui lòng nhập lại nha."
        if result_class == 1:
            # Auto-route penalty queries to NĐ-CP corpus when caller hasn't set a filter
            if metadata_filter is None and _detect_penalty_intent(user_question):
                metadata_filter = _PENALTY_DOC_FILTER
            return self.retrieval.query(user_question, history, metadata_filter)
        return "Câu hỏi không liên quan đến giao thông đường bộ. Bạn vui lòng hỏi câu khác nha."
