import os
import pickle
import time

import faiss

import rag_pipeline as rag


print("=== PHASE 4 GROUNDING RECHECK ===")
index_path, corpus_path, _ = rag._cache_paths(rag.CFG.cache_dir)
if not os.path.exists(index_path) or not os.path.exists(corpus_path):
    raise RuntimeError("The verified Phase 2/3 index cache is required for Phase 4 evidence.")

index = faiss.read_index(index_path)
with open(corpus_path, "rb") as corpus_file:
    corpus = pickle.load(corpus_file)

embedding_model = rag.SentenceTransformer(rag.CFG.embedding_model)
bm25 = rag.build_bm25(corpus)
generator = rag.load_qwen_generator()
if generator is None:
    raise RuntimeError("Qwen did not load; no extractive fallback is accepted for Phase 4 evidence.")

query = "What is the functionality of the 7SR18 Solkor protection relay?"
print(f"QUERY 5 GROUNDED RECHECK: {query}")
retrieved_chunks = rag.retrieve_hybrid(query, embedding_model, index, bm25, corpus)[:3]
for rank, chunk in enumerate(retrieved_chunks, start=1):
    page = f", p.{chunk['page']}" if chunk.get("page") else ""
    print(f"  Rank {rank} | score={chunk['score']:.4f} | [{chunk['source']}{page}]")

started_at = time.perf_counter()
answer = rag.generate_answer(query, retrieved_chunks, generator=generator)
elapsed_seconds = time.perf_counter() - started_at
print(f"GENERATION TIME: {elapsed_seconds:.2f} seconds")
print("FULL GENERATED ANSWER:")
print(answer)
print("=== PHASE 4 GROUNDING RECHECK SUCCESSFUL ===")
