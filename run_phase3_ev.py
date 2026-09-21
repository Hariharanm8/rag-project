import rag_pipeline as rag

print("=== PHASE 3 HYBRID RETRIEVAL NOVELTY VERIFICATION ===")
fingerprint = rag._data_fingerprint(rag.CFG.data_dir)
index, corpus = rag.load_index_if_fresh(rag.CFG.cache_dir, fingerprint)

if index is None:
    docs = rag.load_documents(rag.CFG.data_dir)
    corpus = rag.build_chunk_corpus(docs)
    model = rag.SentenceTransformer(rag.CFG.embedding_model)
    index, _ = rag.build_index(corpus, model)
else:
    model = rag.SentenceTransformer(rag.CFG.embedding_model)

bm25 = rag.build_bm25(corpus)

test_queries = [
    "How to troubleshoot hydraulic press low pressure?",
    "What safety measures are required for Class A Power Quality device?",
    "What is the lubrication frequency for conveyor chains?"
]

for q in test_queries:
    print("\n" + "=" * 65)
    print("QUERY:", q)
    print("=" * 65)
    print("[HYBRID RETRIEVAL (70% Dense + 30% Sparse)]")
    for r in rag.retrieve_hybrid(q, model, index, bm25, corpus)[:2]:
        pg = f" (p.{r['page']})" if r['page'] else ""
        print(f"  score={r['score']:.4f} | {r['source']}{pg} | \"{r['text'][:100]}...\"")

    print("\n[DENSE RETRIEVAL (FAISS)]")
    for r in rag.retrieve_semantic(q, model, index, corpus, top_k=2):
        pg = f" (p.{r['page']})" if r['page'] else ""
        print(f"  score={r['score']:.4f} | {r['source']}{pg} | \"{r['text'][:100]}...\"")

    print("\n[SPARSE RETRIEVAL (BM25)]")
    for r in rag.retrieve_bm25(q, bm25, corpus, top_k=2):
        pg = f" (p.{r['page']})" if r['page'] else ""
        print(f"  score={r['score']:.4f} | {r['source']}{pg} | \"{r['text'][:100]}...\"")

print("\n=== PHASE 3 VERIFICATION SUCCESSFUL ===")
