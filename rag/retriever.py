from typing import Optional

from rag.embeddings import embed_text
from rag.ingestion import get_or_create_collection
from config.settings import TOP_K_RETRIEVAL


def retrieve(query: str, source_type: str, top_k: int = TOP_K_RETRIEVAL) -> list[dict]:
    collection = get_or_create_collection(source_type)
    query_embedding = embed_text(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count() or 1),
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text": doc,
            "chunk_type": meta.get("chunk_type", "text"),
            "file_name": meta.get("file_name", ""),
            "page_number": meta.get("page_number", 0),
            "source_type": meta.get("source_type", source_type),
            "image_data": meta.get("image_data", ""),
            "relevance_score": round(1 - dist, 4),
        })

    return chunks


def retrieve_multi(query: str, source_types: list[str], top_k: int = TOP_K_RETRIEVAL) -> dict[str, list[dict]]:
    return {src: retrieve(query, src, top_k) for src in source_types}
