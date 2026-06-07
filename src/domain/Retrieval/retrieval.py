# Retrieval/retrieval.py
from __future__ import annotations

from typing import Any

from langchain_aws import ChatBedrock
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import config

try:
    from observability.tracing import get_callbacks

    _CALLBACKS = get_callbacks()
except Exception:
    _CALLBACKS = []

qa_prompt = ChatPromptTemplate.from_template(
    "Bạn là trợ lý ảo giúp trả lời các câu hỏi về luật giao thông đường bộ Việt Nam.\n"
    "Chỉ trả lời dựa trên ngữ cảnh được cung cấp, không thêm thông tin bên ngoài.\n\n"
    "QUY TẮC BẮT BUỘC:\n"
    "- Số tiền phạt, nồng độ cồn, tốc độ km/h, thời hạn tước bằng: TRÍCH NGUYÊN VĂN từ ngữ cảnh, KHÔNG làm tròn hoặc diễn giải lại.\n"
    "- Nếu ngữ cảnh không có số liệu cụ thể: trả lời 'Không tìm thấy thông tin cụ thể trong dữ liệu.'\n"
    "- Luôn kèm trích dẫn điều khoản (ví dụ: Điều 7 Nghị định 168/2024/NĐ-CP).\n\n"
    "Ngữ cảnh:\n---------------------\n{context}\n---------------------\n"
    "Lịch sử hội thoại:\n{history}\n\n"
    "Câu hỏi: {question}\n"
    "Câu trả lời:"
)


def _format_docs(docs: list[Document]) -> str:
    return "\n\n".join(
        f"Tiêu đề: {d.metadata.get('title', 'Không có tiêu đề')}\nNội dung: {d.page_content}"
        for d in docs
    )


def _build_chroma_filter(metadata_filter: dict[str, Any] | None) -> dict | None:
    """Build Chroma $and where-filter from metadata key-value pairs."""
    if not metadata_filter:
        return None
    conditions = []
    for key, value in metadata_filter.items():
        if isinstance(value, list):
            conditions.append({key: {"$in": value}})
        else:
            conditions.append({key: {"$eq": value}})
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


_EXACT_KEYWORDS = [
    "phạt", "bị phạt", "tiền phạt", "xử phạt", "mức phạt",
    "km/h", "miligam", "nồng độ", "tước", "trừ điểm",
    "bao nhiêu tiền", "phạt bao nhiêu", "bao nhiêu năm", "bao nhiêu tháng",
]


def _get_hybrid_weights(question: str) -> tuple[float, float]:
    """BM25-dominant for exact-value queries; vector-dominant otherwise."""
    q = question.lower()
    if any(kw in q for kw in _EXACT_KEYWORDS):
        return 0.65, 0.35  # BM25 dominant
    return config.BM25_WEIGHT, config.VECTOR_WEIGHT


class _CohereReranker:
    """LangChain-compatible document compressor using Cohere Rerank v3 on Bedrock."""

    def __init__(self, top_n: int = 5):
        self.top_n = top_n
        import boto3
        self._client = boto3.client("bedrock-runtime", region_name=config.AWS_REGION)

    def compress_documents(self, documents: list[Document], query: str, callbacks=None) -> list[Document]:
        import json as _json
        if not documents:
            return []
        body = _json.dumps({
            "query": query,
            "documents": [d.page_content[:2048] for d in documents],
            "top_n": min(self.top_n, len(documents)),
            "api_version": 2,
        })
        resp = self._client.invoke_model(modelId="cohere.rerank-v3-5:0", body=body)
        results = _json.loads(resp["body"].read())["results"]
        return [documents[r["index"]] for r in results]


def _build_retriever(vectorstore, documents, metadata_filter: dict[str, Any] | None = None, question: str = ""):
    search_kwargs: dict[str, Any] = {"k": config.SIMILARITY_TOP_K}
    chroma_filter = _build_chroma_filter(metadata_filter)
    if chroma_filter:
        search_kwargs["filter"] = chroma_filter

    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)

    if config.ENABLE_HYBRID and documents:
        from langchain.retrievers import EnsembleRetriever
        from langchain_community.retrievers import BM25Retriever

        # Filter BM25 documents to match metadata filter
        filtered_docs = documents
        if metadata_filter:
            filtered_docs = [
                d for d in documents
                if all(
                    (str(d.metadata.get(k)) in [str(v) for v in val]) if isinstance(val, list)
                    else str(d.metadata.get(k)) == str(val)
                    for k, val in metadata_filter.items()
                )
            ]
        if filtered_docs:
            bm25_w, vec_w = _get_hybrid_weights(question)
            bm25 = BM25Retriever.from_documents(filtered_docs)
            bm25.k = config.SIMILARITY_TOP_K
            retriever = EnsembleRetriever(
                retrievers=[bm25, retriever],
                weights=[bm25_w, vec_w],
            )

    return retriever


class Retrieval:
    """Hybrid RAG (BM25 + vector) với Bedrock Claude, hỗ trợ metadata filtering + graph expansion."""

    def __init__(self, vectorstore, documents=None):
        self.vectorstore = vectorstore
        self.documents = documents
        self.retriever = _build_retriever(vectorstore, documents, question="")
        self.llm = ChatBedrock(model_id=config.LLM_MODEL_ID, region_name=config.AWS_REGION)
        self.chain = (qa_prompt | self.llm | StrOutputParser()).with_config({"callbacks": _CALLBACKS})
        self._reranker = _CohereReranker(top_n=config.RERANK_TOP_N)

        # Build legal relationship graph for context expansion
        self.graph = None
        if documents:
            from Retrieval.legal_graph import LegalGraph
            self.graph = LegalGraph(documents)

    def retrieve(self, question: str, metadata_filter: dict[str, Any] | None = None, expand_graph: bool = True) -> list[Document]:
        """Retrieve documents with optional metadata filtering and graph-based expansion."""
        if metadata_filter:
            retriever = _build_retriever(self.vectorstore, self.documents, metadata_filter, question)
        else:
            retriever = _build_retriever(self.vectorstore, self.documents, question=question)
        docs = retriever.invoke(question)

        # Disable graph expansion when filtered to NĐ-CP (penalty corpus):
        # graph refs are bare strings with no doc prefix → expand adds noise, not signal
        nd_filter = metadata_filter and metadata_filter.get("doc_id") == "168/2024/NĐ-CP"
        if expand_graph and not nd_filter and self.graph and docs:
            related = self.graph.get_related_docs(docs, max_hops=1, max_expand=3)
            docs = docs + related

        if config.ENABLE_RERANK and docs:
            docs = self._reranker.compress_documents(docs, question)
        return docs

    def query_with_context(
        self, question: str, history: str = "", metadata_filter: dict[str, Any] | None = None
    ) -> tuple[str, list[Document]]:
        docs = self.retrieve(question, metadata_filter, expand_graph=True)
        if not docs:
            return "Không tìm thấy thông tin liên quan.", []
        answer = self.chain.invoke(
            {"context": _format_docs(docs), "history": history or "Không có", "question": question}
        )
        return answer, docs

    def query(self, question: str, history: str = "", metadata_filter: dict[str, Any] | None = None) -> str:
        return self.query_with_context(question, history, metadata_filter)[0]
