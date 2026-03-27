import json
import pickle
from pathlib import Path
from typing import List
from sqlalchemy import text
from rank_bm25 import BM25Okapi
from src.ingestion.database import engine
from src.ingestion.chunker import Chunk

BM25_INDEX_PATH = "models/bm25_index.pkl"
BM25_CORPUS_PATH = "models/bm25_corpus.pkl"


def store_chunks(chunks: List[Chunk],
                 embeddings: List[List[float]]) -> None:
    """Store chunks and their embeddings in PostgreSQL."""
    assert len(chunks) == len(embeddings), \
        "Chunks and embeddings must have the same length"

    with engine.connect() as conn:
        for chunk, embedding in zip(chunks, embeddings):
            conn.execute(text("""
                INSERT INTO document_chunks
                    (content, embedding, source_file,
                     chunk_index, doc_type, metadata)
                VALUES
                    (:content, :embedding, :source_file,
                     :chunk_index, :doc_type, :metadata)
            """), {
                "content": chunk.content,
                "embedding": str(embedding),  # pgvector accepts list as string
                "source_file": chunk.source_file,
                "chunk_index": chunk.chunk_index,
                "doc_type": chunk.doc_type,
                "metadata": json.dumps(chunk.metadata)
            })

        conn.commit()

    print(f"Stored {len(chunks)} chunks in PostgreSQL.")


def build_bm25_index(chunks: List[Chunk]) -> None:
    """Build and serialize BM25 index from chunks."""
    Path("models").mkdir(exist_ok=True)

    corpus = [chunk.content.lower().split() for chunk in chunks]
    bm25 = BM25Okapi(corpus)

    # Save the index and the original chunks for retrieval
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25, f)

    # Save chunk contents mapped to their DB-insertion order
    corpus_meta = [
        {"content": chunk.content,
         "source_file": chunk.source_file,
         "doc_type": chunk.doc_type,
         "chunk_index": chunk.chunk_index}
        for chunk in chunks
    ]
    with open(BM25_CORPUS_PATH, "wb") as f:
        pickle.dump(corpus_meta, f)

    print(f"BM25 index built and saved to {BM25_INDEX_PATH}")
    print(f"BM25 corpus saved to {BM25_CORPUS_PATH}")