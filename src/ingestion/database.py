import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create tables and indexes."""
    with engine.connect() as conn:
        # Enable pgvector extension
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

        # Create chunks table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id          SERIAL PRIMARY KEY,
                content     TEXT NOT NULL,
                embedding   vector(384),
                source_file TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                doc_type    TEXT,
                metadata    JSONB DEFAULT '{}',
                created_at  TIMESTAMP DEFAULT NOW()
            )
        """))

        # Create ivfflat index for fast approximate nearest neighbor search
        # Note: needs at least 100 rows before this becomes useful
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_chunks_embedding
            ON document_chunks
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100)
        """))

        # Create index on source_file for filtered queries
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_chunks_source
            ON document_chunks (source_file)
        """))

        conn.commit()
        print("Database initialized successfully.")


if __name__ == "__main__":
    init_db()