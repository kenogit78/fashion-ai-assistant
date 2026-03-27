#!/usr/bin/env python3
"""
Main ingestion pipeline.
Usage: python ingest.py
"""
import os
import time
from dotenv import load_dotenv

load_dotenv()

from src.ingestion.database import init_db
from src.ingestion.loader import load_directory
from src.ingestion.chunker import chunk_documents
from src.ingestion.embedder import embed_texts
from src.ingestion.storer import store_chunks, build_bm25_index

DATA_DIR = "data/raw"
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "128"))


def main():
    print("=" * 60)
    print("FASHION AI ASSISTANT — INGESTION PIPELINE")
    print("=" * 60)

    # Step 1: Initialize database
    print("\n[1/5] Initializing database...")
    init_db()

    # Step 2: Load documents
    print(f"\n[2/5] Loading documents from {DATA_DIR}...")
    start = time.time()
    docs = load_directory(DATA_DIR)
    print(f"      Done in {time.time() - start:.1f}s")

    if not docs:
        print("ERROR: No documents found. Check your data/raw/ directory.")
        return

    # Step 3: Chunk documents
    print(f"\n[3/5] Chunking documents "
          f"(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    start = time.time()
    chunks = chunk_documents(docs, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"      Done in {time.time() - start:.1f}s")

    # Step 4: Embed chunks
    print(f"\n[4/5] Embedding {len(chunks)} chunks...")
    print("      This may take several minutes on first run "
          "(model download + encoding)...")
    start = time.time()
    texts = [chunk.content for chunk in chunks]
    embeddings = embed_texts(texts, batch_size=64, show_progress=True)
    print(f"      Done in {time.time() - start:.1f}s")

    # Step 5: Store in PostgreSQL + build BM25 index
    print(f"\n[5/5] Storing chunks and building BM25 index...")
    start = time.time()
    store_chunks(chunks, embeddings)
    build_bm25_index(chunks)
    print(f"      Done in {time.time() - start:.1f}s")

    print("\n" + "=" * 60)
    print("INGESTION COMPLETE")
    print(f"  Documents loaded:  {len(docs)}")
    print(f"  Chunks created:    {len(chunks)}")
    print(f"  Chunks embedded:   {len(embeddings)}")
    print("=" * 60)


if __name__ == "__main__":
    main()