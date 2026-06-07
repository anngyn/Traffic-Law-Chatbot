# Retrieval/database.py
from __future__ import annotations

import json
import os

from langchain_aws import BedrockEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

import config


def _make_embeddings(input_type: str | None = None) -> BedrockEmbeddings:
    """Build BedrockEmbeddings, injecting input_type for Cohere models."""
    kwargs: dict = {}
    if input_type and "cohere" in config.EMBEDDING_MODEL_ID:
        kwargs["model_kwargs"] = {"input_type": input_type}
    return BedrockEmbeddings(
        model_id=config.EMBEDDING_MODEL_ID,
        region_name=config.AWS_REGION,
        **kwargs,
    )


class ChromaVectorStoreManager:
    """Quản lý Vector Store (ChromaDB) qua LangChain với embedding Bedrock."""

    def __init__(self, collection_name: str = config.COLLECTION_NAME, data_folder: str = config.DATA_FOLDER):
        # search_query for retrieval; search_document used during store()
        self.embeddings = _make_embeddings("search_query")
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=os.path.join(data_folder, "chroma_db"),
        )

    def load_documents(self, processed_json_file: str) -> list[Document]:
        """Load documents từ file JSON đã được tiền xử lý (field-level metadata)."""
        with open(processed_json_file, encoding="utf-8") as f:
            data = json.load(f)

        documents = []
        for item in data:
            metadata = {}
            for key, value in item.items():
                if key == "content":
                    continue
                if isinstance(value, list):  # Chroma metadata không nhận list
                    value = ", ".join(map(str, value))
                metadata[key] = "" if value is None else value
            documents.append(Document(page_content=item.get("content", ""), metadata=metadata))

        print(f"Loaded {len(documents)} documents from {processed_json_file}.")
        return documents

    # Cohere hard limit per text
    _COHERE_MAX_CHARS = 2048

    def store(self, documents: list[Document]) -> None:
        """Embed và lưu documents vào Chroma (tự động persist)."""
        index_emb = _make_embeddings("search_document")
        metadatas = [d.metadata for d in documents]
        full_texts = [d.page_content for d in documents]

        # Truncate for embedding only; full text still stored in Chroma document field
        embed_texts = [t[:self._COHERE_MAX_CHARS] for t in full_texts]

        vectors = index_emb.embed_documents(embed_texts)
        self.vectorstore._collection.add(
            embeddings=vectors,
            documents=full_texts,   # store full content for LLM context
            metadatas=metadatas,
            ids=[str(i) for i in range(len(documents))],
        )
        print(f"Stored {len(documents)} documents into ChromaDB.")

    def count_nodes(self) -> int:
        """Đếm số lượng nodes hiện có trong collection."""
        return self.vectorstore._collection.count()

    def delete_collection(self):
        """Xóa collection hiện tại."""
        self.vectorstore.delete_collection()
