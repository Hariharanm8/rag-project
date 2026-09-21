"""
verify_step3.py
================
Run this directly (no AI agent needed):  python -u verify_step3.py

Checks the 3 critical demo queries against whatever index is currently
loaded/cached in this project folder, using the existing rag_pipeline.py
functions. Paste the output back to Claude for interpretation.
"""
import rag_pipeline as rag

print("=== STEP 3 VERIFICATION (102-document corpus) ===\n")

# Load whatever index is currently cached/built
fingerprint = rag._data_fingerprint(rag.CFG.data_dir)
index, corpus = rag.load_index_if_fresh(rag.CFG.cache_dir, fingerprint)

if index is None:
    print("ERROR: No fresh cache found for the current data/ folder.")
    print("This means the rebuild either hasn't finished, or something is")
    print("still mismatched. Do not proceed -- check the rebuild evidence")
    print("file first.")
    raise SystemExit(1)

print(f"Loaded index: {index.ntotal} vectors, {len(corpus)} passages\n")

model = rag.SentenceTransformer(rag.CFG.embedding_model)
bm25 = rag.build_bm25(corpus)

queries = [
    ("SIRIUS 3RM1 terminals", "How to connect screw-type terminals on a SIRIUS 3RM1 motor starter?"),
    ("SICAM Q200 / Class A Power Quality (CRITICAL - check for divergence)", "What safety measures are required for a Class A Power Quality device?"),
    ("7SR18 refusal case", "[insert the same specific narrow 7SR18 detail question used before]"),
]

for label, q in queries:
    print("=" * 70)
    print(f"[{label}]")
    print(f"QUERY: {q}")
    print("=" * 70)

    print("\n[HYBRID]")
    for r in rag.retrieve_hybrid(q, model, index, bm25, corpus)[:2]:
        pg = f" (p.{r['page']})" if r.get('page') else ""
        print(f"  score={r['score']:.4f} | {r['source']}{pg}")

    print("\n[DENSE / FAISS]")
    for r in rag.retrieve_semantic(q, model, index, corpus, top_k=2):
        pg = f" (p.{r['page']})" if r.get('page') else ""
        print(f"  score={r['score']:.4f} | {r['source']}{pg}")

    print("\n[SPARSE / BM25]")
    for r in rag.retrieve_bm25(q, bm25, corpus, top_k=2):
        pg = f" (p.{r['page']})" if r.get('page') else ""
        print(f"  score={r['score']:.4f} | {r['source']}{pg}")
    print()

print("=== DONE -- paste this full output back for review ===")