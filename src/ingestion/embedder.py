import os
from typing import List
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))

# Load once at module level — expensive to reload
_model = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"Loading embedding model: {EMBEDDING_MODEL}")
        _model = SentenceTransformer(EMBEDDING_MODEL)
        print("Embedding model loaded.")
    return _model


def embed_texts(texts: List[str],
                batch_size: int = 64,
                show_progress: bool = True) -> List[List[float]]:
    """
    Embed a list of texts. Returns list of 384-dim float vectors.
    Processes in batches to avoid memory issues.
    """
    model = get_model()

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True  # Cosine similarity works better normalized
    )

    return embeddings.tolist()


def embed_query(query: str) -> List[float]:
    """Embed a single query string."""
    model = get_model()
    embedding = model.encode(
        [query],
        normalize_embeddings=True
    )
    return embedding[0].tolist()