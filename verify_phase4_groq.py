"""One-query synchronous verification of the Groq citation-generation backend."""
import os
import pickle
import time

import faiss
import rag_pipeline as rag

print("=== PHASE 4 GROQ GENERATION WITH CITATIONS ===")
print(f"GROQ_API_KEY present: {bool(os.environ.get('GROQ_API_KEY'))}")
print(f"Generator backend: {rag.CFG.generator_backend}")
print(f"Generator model: {rag.CFG.generator_model}")

index_path, corpus_path, _ = rag._cache_paths(rag.CFG.cache_dir)
index = faiss.read_index(index_path)
with open(corpus_path, "rb") as corpus_file:
    corpus = pickle.load(corpus_file)

embedding_model = rag.SentenceTransformer(rag.CFG.embedding_model)
bm25 = rag.build_bm25(corpus)
generator = rag.load_generator()
if generator is None:
    raise RuntimeError("Groq generator did not initialize.")

query = "What is the lubrication frequency for conveyor chains?"
chunks = rag.retrieve_hybrid(query, embedding_model, index, bm25, corpus)[:3]
print(f"QUERY: {query}")
for rank, chunk in enumerate(chunks, start=1):
    page = f", p.{chunk['page']}" if chunk.get("page") else ""
    print(f"  Rank {rank} | score={chunk['score']:.4f} | [{chunk['source']}{page}]")

started_at = time.perf_counter()
print("FULL GENERATED ANSWER:")
print(rag.generate_answer(query, chunks, generator))
print(f"GENERATION TIME: {time.perf_counter() - started_at:.2f} seconds")
print("=== PHASE 4 GROQ VERIFICATION SUCCESSFUL ===")
