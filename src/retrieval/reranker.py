import os
from typing import List
from src.retrieval.retriever import RetrievedChunk

_reranker_model = None
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _load_reranker():
    """Load cross-encoder model once."""
    global _reranker_model
    if _reranker_model is None:
        from sentence_transformers.cross_encoder import CrossEncoder
        print(f"Loading reranker model: {RERANKER_MODEL}")
        _reranker_model = CrossEncoder(RERANKER_MODEL)
        print("Reranker loaded.")
    return _reranker_model


def rerank(query: str,
           chunks: List[RetrievedChunk],
           top_k: int = 5) -> List[RetrievedChunk]:
    """
    Rerank retrieved chunks using cross-encoder.
    Takes top-k from hybrid retrieval (usually 20),
    returns top_k most relevant after reranking.
    """
    if not chunks:
        return []

    model = _load_reranker()

    # Cross-encoder scores query+chunk pairs together
    pairs = [[query, chunk.content] for chunk in chunks]
    scores = model.predict(pairs)

    # Attach reranker scores and sort
    scored_chunks = list(zip(chunks, scores))
    scored_chunks.sort(key=lambda x: x[1], reverse=True)

    reranked = []
    for chunk, score in scored_chunks[:top_k]:
        reranked.append(RetrievedChunk(
            chunk_id=chunk.chunk_id,
            content=chunk.content,
            source_file=chunk.source_file,
            doc_type=chunk.doc_type,
            chunk_index=chunk.chunk_index,
            score=float(score),
            retrieval_method="hybrid_reranked",
            metadata=chunk.metadata
        ))

    return reranked