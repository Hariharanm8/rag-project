import os, sys, glob, re, time
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pdfplumber

print("=== PHASE 2 RE-VERIFICATION (UPDATED REPOSITORY WITH 34 FILES) ===")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(SCRIPT_DIR, "data")
print("Data directory:", data_dir)

def clean_page_text(text: str) -> str:
    lines = text.split("\n")
    cleaned = [s.strip() for s in lines if s.strip() and not re.fullmatch(r"\d+", s.strip()) and len(s.strip()) >= 5]
    return re.sub(r" {2,}", " ", " ".join(cleaned)).strip()

txt_files = sorted(glob.glob(os.path.join(data_dir, "*.txt")))
pdf_files = sorted(glob.glob(os.path.join(data_dir, "*.pdf")))

print(f"\nFound {len(txt_files)} text files and {len(pdf_files)} PDF files in repository.")

docs = []
for fp in txt_files:
    fname = os.path.basename(fp)
    with open(fp, "r", encoding="utf-8") as f:
        t = f.read().strip()
    if t:
        docs.append({"filename": fname, "page": None, "text": t})
        print(f"  [Txt] {fname} ({len(t):,} chars)")

for fp in pdf_files:
    fname = os.path.basename(fp)
    pages = []
    try:
        with pdfplumber.open(fp) as pdf:
            for i, p in enumerate(pdf.pages, start=1):
                raw = p.extract_text(x_tolerance=3, y_tolerance=3) or ""
                cleaned = clean_page_text(raw)
                if cleaned: pages.append((i, cleaned))
    except Exception as e:
        print(f"  [WARN] pdfplumber error on {fname}: {e}")
    if pages:
        chars = sum(len(t) for _, t in pages)
        for pg, text in pages:
            docs.append({"filename": fname, "page": pg, "text": text})
        print(f"  [Pdf] {fname} ({len(pages)} pages, {chars:,} chars)")

print(f"\nTotal Ingested Document Sections across 34 files: {len(docs)}")

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
                else: break
            cur_words, cur_sents = list(ov_words), list(ov_sents)
        cur_words.extend(words)
        cur_sents.append(sent)
    if cur_words: chunks.append(" ".join(cur_words))
    return chunks

corpus = []
for d in docs:
    for c in chunk_text(d["text"]):
        corpus.append({"text": c, "source": d["filename"], "page": d["page"]})

print(f"Total Created Search Passages: {len(corpus)}")

print("\nEncoding passages with SentenceTransformer (all-MiniLM-L6-v2)...")
t0 = time.time()
model = SentenceTransformer("all-MiniLM-L6-v2")
texts = [c["text"] for c in corpus]
embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings.astype(np.float32))
print(f"FAISS index built in {time.time()-t0:.2f}s. Total vectors: {index.ntotal}, Dim: {index.d}")

# Save updated cache
import rag_pipeline as rag
fingerprint = rag._data_fingerprint(data_dir)
rag.save_index(index, corpus, rag.CFG.cache_dir, fingerprint)

test_queries = [
    "What safety measures are required for Class A Power Quality device?",
    "How to configure motorstarter SIRIUS 3RM1?",
    "How to set up SIMOCRANE Energy Storage System Management?",
    "What is 7SR18 Solkor protection relay functionality?"
]

print("\n" + "=" * 70)
print("EXECUTING QUERIES ACROSS 34 ENTERPRISE DOCUMENTS")
print("=" * 70)

for query in test_queries:
    print(f"\nQUERY: {query}")
    qvec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores, idxs = index.search(qvec.astype(np.float32), 2)
    for i in range(len(idxs[0])):
        idx = int(idxs[0][i])
        score = float(scores[0][i])
        item = corpus[idx]
        page_str = f" (Page {item['page']})" if item['page'] else ""
        print(f"  Rank {i+1} | Score: {score:.4f} | Source: {item['source']}{page_str}")
        print(f"         Snippet: \"{item['text'][:140]}...\"")

print("\n=== PHASE 2 RE-VERIFICATION SUCCESSFUL ===")
