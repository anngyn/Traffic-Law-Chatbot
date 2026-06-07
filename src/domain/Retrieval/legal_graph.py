# Retrieval/legal_graph.py — Lightweight graph linking legal entities for context expansion.
from __future__ import annotations

from collections import defaultdict
from typing import Any

from langchain_core.documents import Document


class LegalGraph:
    """In-memory graph of legal relationships: doc -> chuong -> dieu, cross-references, and entity links."""

    def __init__(self, documents: list[Document]):
        # Adjacency: node_id -> set of related node_ids
        self.adj: dict[str, set[str]] = defaultdict(set)
        # node_id -> document index mapping
        self.node_docs: dict[str, list[int]] = defaultdict(list)
        self._documents = documents
        self._build(documents)

    def _node_id(self, meta: dict[str, Any]) -> str:
        """Canonical node id from metadata."""
        doc_id = meta.get("doc_id", "")
        dieu = meta.get("dieu", "")
        khoan = meta.get("khoan", "")
        if khoan:
            return f"{doc_id}:d{dieu}:k{khoan}"
        if dieu:
            return f"{doc_id}:d{dieu}"
        return doc_id

    def _build(self, documents: list[Document]) -> None:
        for idx, doc in enumerate(documents):
            m = doc.metadata
            node = self._node_id(m)
            self.node_docs[node].append(idx)

            doc_id = m.get("doc_id", "")
            chuong = m.get("chuong", "")
            dieu = m.get("dieu", "")

            # Hierarchical edges: doc <-> chuong <-> dieu
            if doc_id:
                self.adj[doc_id].add(node)
                self.adj[node].add(doc_id)
            if chuong and doc_id:
                chuong_node = f"{doc_id}:ch{chuong}"
                self.adj[chuong_node].add(node)
                self.adj[node].add(chuong_node)
                self.adj[doc_id].add(chuong_node)
                self.adj[chuong_node].add(doc_id)

            # Cross-reference edges from 'references' field
            refs = m.get("references", "")
            if refs:
                for ref in (refs.split(", ") if isinstance(refs, str) else refs):
                    ref = ref.strip()
                    if ref:
                        self.adj[node].add(ref)
                        self.adj[ref].add(node)

    def get_related_docs(self, seed_docs: list[Document], max_hops: int = 1, max_expand: int = 5) -> list[Document]:
        """BFS expansion from seed documents through the graph, returning additional related docs."""
        seed_ids = {self._node_id(d.metadata) for d in seed_docs}
        visited = set(seed_ids)
        frontier = set(seed_ids)

        for _ in range(max_hops):
            next_frontier: set[str] = set()
            for nid in frontier:
                for neighbor in self.adj.get(nid, set()):
                    if neighbor not in visited:
                        next_frontier.add(neighbor)
                        visited.add(neighbor)
            frontier = next_frontier

        # Collect documents from expanded nodes (exclude seeds)
        expanded_indices: list[int] = []
        for nid in visited - seed_ids:
            expanded_indices.extend(self.node_docs.get(nid, []))

        # Deduplicate and limit
        seen: set[int] = {i for nid in seed_ids for i in self.node_docs.get(nid, [])}
        result: list[Document] = []
        for i in expanded_indices:
            if i not in seen:
                seen.add(i)
                result.append(self._documents[i])
                if len(result) >= max_expand:
                    break
        return result
