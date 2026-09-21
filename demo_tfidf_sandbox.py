"""
Sandbox-only verification script.
This environment blocks downloads from huggingface.co, so we can't pull
the real sentence-transformers model here. This script proves the
chunking + retrieval LOGIC works correctly using TF-IDF vectors instead
(scikit-learn, no download required). On your own machine or Google
Colab (which has normal internet access), use rag_pipeline.py instead -
that one uses real semantic embeddings and is your actual project code.
"""
import glob
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from rag_pipeline import load_documents, build_chunk_corpus, keyword_search

# Resolve paths relative to this script's location, not the CWD
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

documents = load_documents(os.path.join(SCRIPT_DIR, "data"))
corpus = build_chunk_corpus(documents)
print(f"Loaded {len(documents)} documents -> {len(corpus)} chunks\n")

texts = [c["text"] for c in corpus]
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(texts)

test_queries = [
    "What should I do if the press doesn't build up pressure?",
    "How often do we need to oil the conveyor chain?",
    "What is the most common reason incoming steel gets rejected?",
]

for query in test_queries:
    print("=" * 70)
    print(f"QUERY: {query}")
    print("=" * 70)

    query_vec = vectorizer.transform([query])
    sims = cosine_similarity(query_vec, tfidf_matrix)[0]
    top_idx = sims.argsort()[::-1][:2]

    print("\n[TF-IDF Retrieval - stand-in for semantic search in this sandbox]")
    for idx in top_idx:
        print(f"  score={sims[idx]:.3f}  source={corpus[idx]['source']}")
        print(f"  \"{corpus[idx]['text'][:180]}...\"")

    print("\n[Keyword Overlap Baseline]")
    for r in keyword_search(query, corpus, top_k=2):
        print(f"  overlap={r['score']}  source={r['source']}")
        print(f"  \"{r['text'][:180]}...\"")
    print()
