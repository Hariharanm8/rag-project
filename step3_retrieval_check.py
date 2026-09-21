"""
Step 3: Direct retrieval comparison for the SICAM Q200 Class A query
on the new 15,640-passage full corpus.
Run: python -u step3_retrieval_check.py > step3_evidence.txt 2>&1
"""
import os
os.environ["RAG_ENABLE_OCR_FALLBACK"] = "0"

import rag_pipeline as rag

print("=== STEP 3: RETRIEVAL COMPARISON ON FULL CORPUS ===", flush=True)
print(f"Loading cache from {rag.CFG.cache_dir} ...", flush=True)

QUERY = "What safety measures are required for a Class A Power Quality device?"

fingerprint = rag._data_fingerprint(rag.CFG.data_dir)
index, corpus = rag.load_index_if_fresh(rag.CFG.cache_dir, fingerprint)
if index is None:
    print("ERROR: Cache miss! Cannot proceed without the full-corpus cache.", flush=True)
    raise SystemExit(1)

print(f"Cache loaded: {index.ntotal} vectors, {len(corpus)} passages", flush=True)

model = rag.SentenceTransformer(rag.CFG.embedding_model)
bm25  = rag.build_bm25(corpus)

print(f"\nQuery: {QUERY!r}", flush=True)

# --- Dense (semantic) ---
print("\n--- SEMANTIC (dense FAISS) top-5 ---", flush=True)
sem_results = rag.retrieve_semantic(QUERY, model, index, corpus, top_k=5)
for i, r in enumerate(sem_results, 1):
    src  = r.get("source", "?")
    page = r.get("page", "?")
    score = r.get("score", 0.0)
    snippet = r.get("text", "")[:120].replace("\n", " ").encode("ascii", "replace").decode("ascii")
    print(f"  #{i}  score={score:.4f}  src={src}  page={page}", flush=True)
    print(f"       snippet: {snippet!r}", flush=True)

# --- BM25 (keyword) ---
print("\n--- BM25 (keyword) top-5 ---", flush=True)
bm25_results = rag.retrieve_bm25(QUERY, bm25, corpus, top_k=5)
for i, r in enumerate(bm25_results, 1):
    src  = r.get("source", "?")
    page = r.get("page", "?")
    score = r.get("score", 0.0)
    snippet = r.get("text", "")[:120].replace("\n", " ").encode("ascii", "replace").decode("ascii")
    print(f"  #{i}  score={score:.4f}  src={src}  page={page}", flush=True)
    print(f"       snippet: {snippet!r}", flush=True)

# --- Hybrid ---
print("\n--- HYBRID (70% dense + 30% BM25) top-5 ---", flush=True)
hybrid_results = rag.retrieve_hybrid(QUERY, model, index, bm25, corpus)[:5]
for i, r in enumerate(hybrid_results, 1):
    src  = r.get("source", "?")
    page = r.get("page", "?")
    score = r.get("score", 0.0)
    snippet = r.get("text", "")[:120].replace("\n", " ").encode("ascii", "replace").decode("ascii")
    print(f"  #{i}  score={score:.4f}  src={src}  page={page}", flush=True)
    print(f"       snippet: {snippet!r}", flush=True)

# --- Summary ---
print("\n=== SUMMARY ===", flush=True)
print("BM25 #1 source:", bm25_results[0].get("source","?") if bm25_results else "none", flush=True)
print("Hybrid #1 source:", hybrid_results[0].get("source","?") if hybrid_results else "none", flush=True)
print("Dense #1 source:", sem_results[0].get("source","?") if sem_results else "none", flush=True)

sem_top2_src   = [r.get("source","?") for r in sem_results[:2]]
bm25_top2_src  = [r.get("source","?") for r in bm25_results[:2]]
hybrid_top2_src = [r.get("source","?") for r in hybrid_results[:2]]
print("\nTop-2 sources — Dense:", sem_top2_src, flush=True)
print("Top-2 sources — BM25:", bm25_top2_src, flush=True)
print("Top-2 sources — Hybrid:", hybrid_top2_src, flush=True)

# Check if BM25 top doc differs from hybrid/dense (the main evidence claim)
bm25_top = bm25_results[0].get("source","?") if bm25_results else "?"
dense_top = sem_results[0].get("source","?") if sem_results else "?"
hybrid_top = hybrid_results[0].get("source","?") if hybrid_results else "?"
diverged = (bm25_top != dense_top) or (bm25_top != hybrid_top)
print(f"\nBM25 divergence from hybrid/dense: {diverged}", flush=True)
if diverged:
    print(f"  Dense top:  {dense_top}", flush=True)
    print(f"  BM25 top:   {bm25_top}", flush=True)
    print(f"  Hybrid top: {hybrid_top}", flush=True)
    print("  -> BM25 divergence example HOLDS (or changed source)", flush=True)
else:
    print("  All three methods agree on #1 source for this query.", flush=True)
    print("  -> Need to find a different divergence example.", flush=True)

print("\n=== STEP 3 COMPLETE ===", flush=True)
