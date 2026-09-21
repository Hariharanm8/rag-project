"""Persistent FastAPI service for the enterprise RAG pipeline.

All heavyweight resources are initialized once during application startup and
reused by every POST /query request. The interactive CLI in rag_pipeline.py is
left unchanged.
"""
# RAG_ENABLE_OCR_FALLBACK must be "0" BEFORE importing rag_pipeline.
# The module-level constant _INGESTION_CACHE_VERSION is computed at import
# time; if OCR is enabled (default "1") it produces a different constant than
# the "0" value used by rebuild_full_corpus.py, so the SHA-256 fingerprints
# disagree and the cache always looks stale — causing a full 40-minute rebuild
# with live OvisOCR2 requests on every server startup.
import os
os.environ.setdefault("RAG_ENABLE_OCR_FALLBACK", "0")

import time
from contextlib import asynccontextmanager
from typing import Optional

import faiss
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import rag_pipeline as rag


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)


class Runtime:
    model = None
    index = None
    corpus = None
    bm25 = None
    reranker = None
    generator = None
    startup_seconds = 0.0


RUNTIME = Runtime()


def _load_cached_or_build():
    """Load a fresh cache, or build and persist the current corpus once.

    OCR is disabled via RAG_ENABLE_OCR_FALLBACK=0 (set at module top, before
    the rag_pipeline import) so the fingerprint matches the one saved by
    rebuild_full_corpus.py.  A cache hit is therefore the normal fast path;
    a rebuild is only triggered when the data directory has genuinely changed.
    """
    fingerprint = rag._data_fingerprint(rag.CFG.data_dir)
    index, corpus = rag.load_index_if_fresh(rag.CFG.cache_dir, fingerprint)
    if index is None:
        if os.environ.get("RAG_ALLOW_STALE_CACHE") == "1":
            # Diagnostic mode only: useful for timing/API smoke tests when a
            # corpus has changed but rebuilding its large PDFs is impractical.
            index = faiss.read_index(os.path.join(rag.CFG.cache_dir, "index.faiss"))
            with open(os.path.join(rag.CFG.cache_dir, "corpus.pkl"), "rb") as handle:
                corpus = pickle.load(handle)
            print(f"[API] WARNING: using stale cache ({index.ntotal} vectors)")
            return index, corpus
        # Cache is genuinely missing or stale (data dir changed).  OCR is
        # already disabled at the module level so no OvisOCR2 requests will
        # be made; pdfplumber is the primary extractor as documented.
        print("[API] Cache miss - rebuilding index (OCR disabled).")
        documents = rag.load_documents(rag.CFG.data_dir)
        corpus = rag.build_chunk_corpus(documents)
        index, _ = rag.build_index(corpus, RUNTIME.model)
        rag.save_index(index, corpus, rag.CFG.cache_dir, fingerprint)
    return index, corpus


@asynccontextmanager
async def lifespan(app: FastAPI):
    started = time.perf_counter()
    RUNTIME.model = rag.SentenceTransformer(rag.CFG.embedding_model)
    RUNTIME.index, RUNTIME.corpus = _load_cached_or_build()
    RUNTIME.bm25 = rag.build_bm25(RUNTIME.corpus)
    RUNTIME.reranker = rag.load_reranker()
    RUNTIME.generator = rag.load_generator()
    RUNTIME.startup_seconds = time.perf_counter() - started
    print(f"[API] startup complete in {RUNTIME.startup_seconds:.3f}s; resources loaded once")
    yield
    # Explicit references make shutdown deterministic without affecting CLI mode.
    RUNTIME.model = RUNTIME.index = RUNTIME.corpus = None
    RUNTIME.bm25 = RUNTIME.reranker = RUNTIME.generator = None


app = FastAPI(title="DocMindAI RAG API", version="1.0", lifespan=lifespan)


@app.get("/health")
def health():
    ready = RUNTIME.model is not None and RUNTIME.index is not None
    return {"status": "ok" if ready else "starting", "startup_seconds": RUNTIME.startup_seconds}


@app.post("/query")
def query(request: QueryRequest):
    if RUNTIME.model is None or RUNTIME.index is None:
        raise HTTPException(status_code=503, detail="RAG runtime is still starting")
    started = time.perf_counter()
    passages = rag.retrieve_hybrid(
        request.question, RUNTIME.model, RUNTIME.index, RUNTIME.bm25, RUNTIME.corpus
    )
    if RUNTIME.reranker and passages:
        passages = rag.rerank(request.question, passages, RUNTIME.reranker)
    answer = rag.generate_answer(request.question, passages, generator=RUNTIME.generator)
    citations = []
    for passage in passages[:rag.CFG.final_top_k]:
        cite = f"[{passage['source']}" + (f", p.{passage['page']}]" if passage.get("page") else "]")
        if cite not in citations:
            citations.append(cite)
    return {
        "question": request.question,
        "answer": answer,
        "citations": citations,
        "passage_count": len(passages),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="127.0.0.1", port=8000, reload=False)
