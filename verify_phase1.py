import os, sys, time
import rag_pipeline as rag

print("=== PHASE 1 REAL MACHINE VERIFICATION ===")
print("Data dir:", rag.CFG.data_dir)
print("Loading documents...")
docs = rag.load_documents(rag.CFG.data_dir)
print(f"Loaded {len(docs)} document sections.")

print("Chunking text...")
corpus = rag.build_chunk_corpus(docs)
print(f"Created {len(corpus)} chunks.")

print("Loading SentenceTransformer model (all-MiniLM-L6-v2)...")
t0 = time.time()
model = rag.SentenceTransformer(rag.CFG.embedding_model)
print(f"Model loaded in {time.time()-t0:.2f}s.")

print("Building FAISS vector index...")
t0 = time.time()
index, embeddings = rag.build_index(corpus, model)
print(f"FAISS index built in {time.time()-t0:.2f}s. Total vectors: {index.ntotal}, dimension: {index.d}")

test_queries = [
    "What should I do if the press does not build up pressure?",
    "How often do we need to oil the conveyor chain?",
    "What is the most common reason incoming steel gets rejected?"
]

for query in test_queries:
    print("\n" + "=" * 60)
    print(f"QUERY: {query}")
    print("=" * 60)
    results = rag.retrieve_semantic(query, model, index, corpus, top_k=2)
    for i, r in enumerate(results, start=1):
        print(f"  [{i}] score={r['score']:.4f}  source={r['source']}  page={r['page']}")
        print(f"      snippet: \"{r['text'][:140]}...\"")

print("\n=== PHASE 1 VERIFICATION SUCCESSFUL ===")
