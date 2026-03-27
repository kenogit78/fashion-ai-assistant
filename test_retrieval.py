"""
Quick retrieval smoke test.
Run: python test_retrieval.py
"""
from src.retrieval.retriever import retrieve

TEST_QUERIES = [
    "lightweight summer dress",
    "denim jacket for men",
    "Gen Z campaign strategy",
    "sustainable cotton materials",
]

for query in TEST_QUERIES:
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}")

    for mode in ["dense", "sparse", "hybrid"]:
        results = retrieve(query, mode=mode, top_k=3)
        print(f"\n  [{mode.upper()}] Top 3 results:")
        for i, r in enumerate(results[:3]):
            preview = r.content[:80].replace('\n', ' ')
            print(f"    {i+1}. [{r.doc_type}] {preview}... (score: {r.score:.4f})")