import os
from typing import List
from dataclasses import dataclass, field
from src.ingestion.loader import Document


@dataclass
class Chunk:
    content: str
    source_file: str
    doc_type: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)


def chunk_document(doc: Document,
                   chunk_size: int = 512,
                   overlap: int = 128) -> List[Chunk]:
    """
    Fixed-size token-approximate chunking with overlap.
    We approximate tokens as words / 0.75 (rough word-to-token ratio).
    """
    # Approximate word count from token target
    words = doc.content.split()
    word_chunk_size = int(chunk_size * 0.75)
    word_overlap = int(overlap * 0.75)

    if len(words) <= word_chunk_size:
        # Document fits in one chunk
        return [Chunk(
            content=doc.content,
            source_file=doc.source_file,
            doc_type=doc.doc_type,
            chunk_index=0,
            metadata={**doc.metadata, "is_full_doc": True}
        )]

    chunks = []
    start = 0
    chunk_idx = 0

    while start < len(words):
        end = min(start + word_chunk_size, len(words))
        chunk_words = words[start:end]
        chunk_content = " ".join(chunk_words)

        # Only keep chunks with meaningful content
        if len(chunk_content.strip()) > 20:
            chunks.append(Chunk(
                content=chunk_content,
                source_file=doc.source_file,
                doc_type=doc.doc_type,
                chunk_index=chunk_idx,
                metadata={
                    **doc.metadata,
                    "word_start": start,
                    "word_end": end,
                    "is_full_doc": False
                }
            ))
            chunk_idx += 1

        # Move forward by chunk_size minus overlap
        start += word_chunk_size - word_overlap

        if end == len(words):
            break

    return chunks


def chunk_documents(docs: List[Document],
                    chunk_size: int = 512,
                    overlap: int = 128) -> List[Chunk]:
    """Chunk a list of documents."""
    all_chunks = []

    for doc in docs:
        chunks = chunk_document(doc, chunk_size, overlap)
        all_chunks.extend(chunks)

    print(f"Created {len(all_chunks)} chunks from {len(docs)} documents")
    return all_chunks