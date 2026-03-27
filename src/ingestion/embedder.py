import os
from typing import List
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
EMBEDDING_DIM = int(os.getenv(
    "OPENAI_EMBEDDING_DIM" if LLM_PROVIDER == "openai" else "EMBEDDING_DIM",
    "1536" if LLM_PROVIDER == "openai" else "384"
))


def embed_texts(texts: List[str],
                batch_size: int = 64,
                show_progress: bool = True) -> List[List[float]]:
    """Embed texts using configured provider."""

    if LLM_PROVIDER == "openai":
  
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

        all_embeddings = []
        # OpenAI recommends batches of up to 2048
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = client.embeddings.create(model=model, input=batch)
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

            if show_progress:
                print(f"  Embedded {min(i + batch_size, len(texts))}"
                      f"/{len(texts)} chunks...")

        return all_embeddings

    else:
        # Local sentence-transformers for Gemini provider

        model_name = os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        model = SentenceTransformer(model_name)
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True
        )
        return embeddings.tolist()


def embed_query(query: str) -> List[float]:
    """Embed a single query."""
    return embed_texts([query], show_progress=False)[0]