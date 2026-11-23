"""Interactively test retrieval/semantic matching against the knowledge base
built by build_knowledge_base.py - no eligibility filter or ranking boost
yet (that's Phase 2, not built), so this shows exactly what raw semantic
similarity alone returns, for both collections.

Usage:
  python scripts/tools/query_kb.py                        # interactive loop
  python scripts/tools/query_kb.py "widow pension for woman"   # one-off query
"""

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")  # Windows console defaults to cp1252, breaks on Devanagari

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # backend/, for app.*
from app.services.vector_store import LocalEmbedder, VectorCollection

TOP_K = 8


def load_collections():
    embedder = LocalEmbedder()
    return VectorCollection("discovery_cards", embedder), VectorCollection("field_chunks", embedder)


def _eligibility_summary(meta):
    parts = []
    for key in ("caste", "gender", "marital_status", "occupation_target_group", "benefit_type_fine"):
        val = meta.get(key)
        if val and val != "all":
            parts.append(f"{key}={val}")
    if meta.get("family_income_ceiling"):
        parts.append(f"income<={meta['family_income_ceiling']}")
    if meta.get("age_min") is not None or meta.get("age_max") is not None:
        parts.append(f"age={meta.get('age_min')}-{meta.get('age_max')}")
    return ", ".join(parts) if parts else "no restriction"


def run_query(cards, chunks, query, k=TOP_K):
    print(f"\n=== DISCOVERY CARDS (semantic-only, top {k}) ===")
    for rank, (doc, score) in enumerate(cards.similarity_search_with_score(query, k=k), 1):
        m = doc.metadata
        print(f"{rank}. [{score:.4f}] {m['scheme_name']} ({m['lang']}) - {_eligibility_summary(m)}")

    print(f"\n=== FIELD CHUNKS (semantic-only, top {k}) ===")
    for rank, (doc, score) in enumerate(chunks.similarity_search_with_score(query, k=k), 1):
        m = doc.metadata
        snippet = doc.page_content[:140].replace("\n", " ")
        print(f"{rank}. [{score:.4f}] {m['scheme_name']} / {m['field_type']} ({m['lang']}): {snippet}...")


def main():
    print("Loading embedding model + knowledge base...")
    cards, chunks = load_collections()

    if len(sys.argv) > 1:
        run_query(cards, chunks, " ".join(sys.argv[1:]))
        return

    print("Interactive mode. Type a query (Hindi/Hinglish/English), or 'quit' to exit.")
    while True:
        try:
            query = input("\nquery> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query or query.lower() in ("quit", "exit"):
            break
        run_query(cards, chunks, query)


if __name__ == "__main__":
    main()
