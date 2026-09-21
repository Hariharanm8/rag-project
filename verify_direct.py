import os, sys, glob, re, time
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

print("=== PHASE 1 REAL MACHINE VERIFICATION ===")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(SCRIPT_DIR, "data")
print("Data directory:", data_dir)

# 1. Load documents
docs = []
for fp in sorted(glob.glob(os.path.join(data_dir, "*.txt"))):
    fname = os.path.basename(fp)
    with open(fp, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if text:
        docs.append({"filename": fname, "page": None, "text": text})
        print(f"  [Txt]  {fname} ({len(text):,} chars)")

print(f"\nLoaded {len(docs)} document sections.")

# 2. Chunking
def chunk_text(text, chunk_size=400, overlap=80):
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    chunks, cur_words, cur_sents = [], [], []
    for sent in sentences:
        words = sent.split()
        if len(cur_words) + len(words) > chunk_size and cur_words:
            chunks.append(" ".join(cur_words))
            ov_words, ov_sents = [], []
            for s in reversed(cur_sents):
                w = s.split()
                if len(ov_words) + len(w) <= overlap:
                    ov_words = w + ov_words
                    ov_sents = [s] + ov_sents
                else:
                    break
            cur_words, cur_sents = list(ov_words), list(ov_sents)
        cur_words.extend(words)
        cur_sents.append(sent)
    if cur_words:
        chunks.append(" ".join(cur_words))
    return chunks

corpus = []
for d in docs:
    for c in chunk_text(d["text"]):
        corpus.append({"text": c, "source": d["filename"], "page": d["page"]})

print(f"Created {len(corpus)} chunk passages.")

# 3. Load embedding model
print("\nLoading SentenceTransformer model (all-MiniLM-L6-v2)...")
t0 = time.time()
model = SentenceTransformer("all-MiniLM-L6-v2")
print(f"Model loaded in {time.time()-t0:.2f}s.")

# 4. Build FAISS index
print("\nBuilding FAISS vector index...")
t0 = time.time()
texts = [c["text"] for c in corpus]
embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings.astype(np.float32))
print(f"FAISS index built in {time.time()-t0:.2f}s. Total vectors: {index.ntotal}, Dim: {index.d}")

# 5. Queries
test_queries = [
    "What should I do if the press does not build up pressure?",
    "How often do we need to oil the conveyor chain?",
    "What is the most common reason incoming steel gets rejected?"
]

print("\n" + "=" * 65)
print("EXECUTING SEMANTIC RETRIEVAL QUERIES")
print("=" * 65)

for query in test_queries:
    print(f"\nQUERY: {query}")
    qvec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores, idxs = index.search(qvec.astype(np.float32), 2)
    for i in range(len(idxs[0])):
        idx = int(idxs[0][i])
        score = float(scores[0][i])
        item = corpus[idx]
        print(f"  Rank {i+1} | Score: {score:.4f} | Source: {item['source']}")
        print(f"         Snippet: \"{item['text'][:150]}...\"")

print("\n=== PHASE 1 VERIFICATION SUCCESSFUL ===")
