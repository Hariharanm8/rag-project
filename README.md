# RAG Foundation Pipeline - Setup Instructions

## What this is
The working "30%" slice of your Review 2: document ingestion, chunking,
semantic embedding, vector storage (FAISS), and retrieval — plus a keyword
search baseline so you can show a real comparison in your Results slide.

## Run it (recommended: Google Colab, it's free and has no setup)
1. Go to https://colab.research.google.com and create a new notebook.
2. Upload the `rag_project` folder (or just `data/` + `rag_pipeline.py`)
   using the folder icon on the left sidebar.
3. In a code cell, run:
   ```
   !pip install sentence-transformers faiss-cpu
   !python rag_pipeline.py
   ```
4. You'll see real semantic retrieval output for the 3 sample queries,
   compared against the keyword baseline.

## Run it locally instead
```
pip install sentence-transformers faiss-cpu
cd rag_project
python rag_pipeline.py
```

## Files
- `rag_pipeline.py` — the real implementation (sentence-transformers + FAISS).
  This is your actual project code — use this in your report/PPT.
- `demo_tfidf_sandbox.py` — a verification-only script I used here to prove
  the chunking/retrieval logic works, since this sandbox blocks model
  downloads. You don't need to submit this one, but the TF-IDF vs keyword
  output is genuinely usable as a rough "baseline comparison" data point
  if you want a second baseline beyond keyword search.
- `data/` — 101 sample SOP/manual documents (hydraulic press, quality
  inspection, conveyor system). Swap these for real documents whenever
  you get access to some — the code doesn't need to change, just point
  `load_documents()` at a folder with more .txt files (or extend it to
  read PDFs with `pypdf`, which I can help you add next).

## What to do next (in order)
1. Run `rag_pipeline.py` on Colab and capture a screenshot of real output
   for your Results slide.
2. Swap in more/real documents if you have any (even 10-15 public PDFs
   on a related topic works — e.g. equipment manuals, government SOPs).
3. Add PDF loading support (ask me — quick addition using `pypdf`).
4. Optionally add generation: feed retrieved chunks into a small free
   model (flan-t5-base via `transformers`, or Ollama if installed) to
   produce an actual cited answer, not just retrieved chunks. This is a
   good "next step" to mention even if you don't finish it before Sept 2.
5. Write up the architecture diagram and methodology text based on this
   actual pipeline (I can help sketch this as a diagram).
