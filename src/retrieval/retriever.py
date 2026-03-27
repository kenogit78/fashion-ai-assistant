import os
import pickle
import json
from typing import List, Optional
from dataclasses import dataclass
from openai import OpenAI
from sqlalchemy import text
from dotenv import load_dotenv
from src.ingestion.embedder import embed_query
from src.ingestion.database import engine

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
BM25_INDEX_PATH = "models/bm25_index.pkl"
BM25_CORPUS_PATH = "models/bm25_corpus.pkl"

# Lazy-loaded globals
_bm25 = None
_bm25_corpus = None


@dataclass
class RetrievedChunk:
    chunk_id: int
    content: str
    source_file: str
    doc_type: str
    chunk_index: int
    score: float
    retrieval_method: str  # 'dense' | 'sparse' | 'hybrid'
    metadata: dict


def _load_bm25():
    """Load BM25 index from disk (lazy, once)."""
    global _bm25, _bm25_corpus

    if _bm25 is None:
        if not os.path.exists(BM25_INDEX_PATH):
            raise FileNotFoundError(
                f"BM25 index not found at {BM25_INDEX_PATH}. "
                "Run python ingest.py first."
            )
        with open(BM25_INDEX_PATH, "rb") as f:
            _bm25 = pickle.load(f)
        with open(BM25_CORPUS_PATH, "rb") as f:
            _bm25_corpus = pickle.load(f)

    return _bm25, _bm25_corpus


def _embed_query(query: str) -> List[float]:
    """Embed query using configured provider."""
    if LLM_PROVIDER == "openai":
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.embeddings.create(
            model=os.getenv(
                "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
            ),
            input=[query]
        )
        return response.data[0].embedding
    else:
        return embed_query(query)


def dense_retrieve(query: str,
                   top_k: int = 20,
                   doc_type_filter: Optional[str] = None
                   ) -> List[RetrievedChunk]:
    """
    Dense retrieval using pgvector cosine similarity.
    Optionally filter by doc_type ('catalog', 'trend_report', 'campaign_brief').
    """
    query_embedding = _embed_query(query)
    embedding_str = str(query_embedding)

    filter_clause = ""
    params = {"embedding": embedding_str, "top_k": top_k}

    if doc_type_filter:
        filter_clause = "AND doc_type = :doc_type"
        params["doc_type"] = doc_type_filter

    sql = text(f"""
        SELECT
            id,
            content,
            source_file,
            doc_type,
            chunk_index,
            metadata,
            1 - (embedding <=> CAST(:embedding AS vector)) AS score
        FROM document_chunks
        WHERE 1=1 {filter_clause}
        ORDER BY embedding <=> CAST(:embedding AS vector)
        LIMIT :top_k
    """)

    with engine.connect() as conn:
        results = conn.execute(sql, params).fetchall()

    return [
        RetrievedChunk(
            chunk_id=row.id,
            content=row.content,
            source_file=row.source_file,
            doc_type=row.doc_type,
            chunk_index=row.chunk_index,
            score=float(row.score),
            retrieval_method="dense",
            metadata=row.metadata if isinstance(row.metadata, dict)
                     else json.loads(row.metadata or "{}")
        )
        for row in results
    ]


def sparse_retrieve(query: str,
                    top_k: int = 20) -> List[RetrievedChunk]:
    """
    Sparse BM25 retrieval.
    Returns top-k chunks ranked by BM25 score.
    """
    bm25, corpus = _load_bm25()

    query_tokens = query.lower().split()
    scores = bm25.get_scores(query_tokens)

    # Get top-k indices sorted by score descending
    top_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:top_k]

    results = []
    for rank, idx in enumerate(top_indices):
        if scores[idx] <= 0:
            continue  # Skip zero-score results

        chunk_meta = corpus[idx]
        results.append(RetrievedChunk(
            chunk_id=idx,          # BM25 uses positional index
            content=chunk_meta["content"],
            source_file=chunk_meta["source_file"],
            doc_type=chunk_meta["doc_type"],
            chunk_index=chunk_meta["chunk_index"],
            score=float(scores[idx]),
            retrieval_method="sparse",
            metadata={}
        ))

    return results


def _reciprocal_rank_fusion(
        dense_results: List[RetrievedChunk],
        sparse_results: List[RetrievedChunk],
        k: int = 60) -> List[RetrievedChunk]:
    """
    Reciprocal Rank Fusion merges two ranked lists.
    RRF score = sum(1 / (k + rank)) across all lists.
    k=60 is the standard default from the original RRF paper.
    """
    rrf_scores: dict[str, float] = {}
    chunk_map: dict[str, RetrievedChunk] = {}

    # Score dense results
    for rank, chunk in enumerate(dense_results):
        key = f"{chunk.source_file}_{chunk.chunk_index}"
        rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (k + rank + 1)
        chunk_map[key] = chunk

    # Score sparse results
    for rank, chunk in enumerate(sparse_results):
        key = f"{chunk.source_file}_{chunk.chunk_index}"
        rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (k + rank + 1)
        if key not in chunk_map:
            chunk_map[key] = chunk

    # Sort by combined RRF score
    sorted_keys = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)

    fused_results = []
    for key in sorted_keys:
        chunk = chunk_map[key]
        fused_results.append(RetrievedChunk(
            chunk_id=chunk.chunk_id,
            content=chunk.content,
            source_file=chunk.source_file,
            doc_type=chunk.doc_type,
            chunk_index=chunk.chunk_index,
            score=rrf_scores[key],
            retrieval_method="hybrid",
            metadata=chunk.metadata
        ))

    return fused_results


def hybrid_retrieve(query: str,
                    top_k: int = 20) -> List[RetrievedChunk]:
    """
    Hybrid retrieval: dense + sparse fused with RRF.
    This is the default retrieval mode.
    """
    dense_results = dense_retrieve(query, top_k=top_k)
    sparse_results = sparse_retrieve(query, top_k=top_k)
    fused = _reciprocal_rank_fusion(dense_results, sparse_results)
    return fused[:top_k]


def retrieve(query: str,
             mode: str = "hybrid",
             top_k: int = 20,
             doc_type_filter: Optional[str] = None) -> List[RetrievedChunk]:
    """
    Unified retrieval interface.
    mode: 'dense' | 'sparse' | 'hybrid'
    """
    if mode == "dense":
        return dense_retrieve(query, top_k, doc_type_filter)
    elif mode == "sparse":
        return sparse_retrieve(query, top_k)
    elif mode == "hybrid":
        return hybrid_retrieve(query, top_k)
    else:
        raise ValueError(f"Unknown retrieval mode: {mode}. "
                         "Use 'dense', 'sparse', or 'hybrid'.")