import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")


class LLMClient:
    """
    Unified LLM client supporting OpenAI and Gemini.
    Switch providers via LLM_PROVIDER env variable.
    """

    def __init__(self):
        self.provider = LLM_PROVIDER

        if self.provider == "openai":
            from openai import OpenAI
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
            self.embedding_model = os.getenv(
                "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
            )
        elif self.provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            self.client = genai.GenerativeModel(
                os.getenv("GEMINI_CHAT_MODEL", "gemini-1.5-flash")
            )
            self.chat_model = os.getenv("GEMINI_CHAT_MODEL", "gemini-1.5-flash")
        else:
            raise ValueError(f"Unknown LLM_PROVIDER: {self.provider}")

    def chat(self,
             system_prompt: str,
             user_message: str,
             temperature: float = 0.3) -> str:
        """Send a chat completion request. Returns response text."""

        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=temperature
            )
            return response.choices[0].message.content

        elif self.provider == "gemini":
            full_prompt = f"{system_prompt}\n\n{user_message}"
            response = self.client.generate_content(full_prompt)
            return response.text

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of texts.
        OpenAI: uses text-embedding-3-small (1536 dims)
        Gemini: falls back to local sentence-transformers
        """
        if self.provider == "openai":
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=texts
            )
            return [item.embedding for item in response.data]

        elif self.provider == "gemini":
            # Gemini embedding API is limited — fall back to local model
            from src.ingestion.embedder import embed_texts
            return embed_texts(texts)

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        return self.embed([query])[0]


# Singleton instance — import this throughout the project
llm = LLMClient()