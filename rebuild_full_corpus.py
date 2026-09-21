"""Time-boxed, synchronous rebuild of the 101-file corpus with OCR disabled."""
import os
import sys
import time

# Must be set before importing rag_pipeline so the cache fingerprint records it.
os.environ["RAG_ENABLE_OCR_FALLBACK"] = "0"

from rag_pipeline import (  # noqa: E402
    CFG,
    SentenceTransformer,
    _data_fingerprint,
    build_chunk_corpus,
    build_index,
    load_documents,
    save_index,
)

TIMEBOX_SECONDS = 180 * 60


def check_time(started: float, stage: str) -> None:
    elapsed = time.monotonic() - started
    if elapsed > TIMEBOX_SECONDS:
        raise TimeoutError(f"180-minute time box exceeded during {stage} ({elapsed / 60:.1f} minutes).")


def main() -> None:
    started = time.monotonic()
    print("=== FULL 101-DOCUMENT REBUILD (OCR DISABLED) ===", flush=True)
    print(f"Time box: {TIMEBOX_SECONDS // 60} minutes", flush=True)
    print("OCR fallback: disabled; insufficient scanned/textless pages are omitted.", flush=True)
    documents = load_documents(CFG.data_dir, progress_interval=20)
    check_time(started, "document ingestion")
    document_count = len({item['filename'] for item in documents})
    print(f"COUNTS AFTER INGESTION: files={document_count}; sections={len(documents)}", flush=True)

    corpus = build_chunk_corpus(documents)
    check_time(started, "chunking")
    print(f"COUNTS AFTER CHUNKING: passages={len(corpus)}", flush=True)

    print(f"Loading embedding model: {CFG.embedding_model}", flush=True)
    model = SentenceTransformer(CFG.embedding_model)
    check_time(started, "embedding-model load")
    print("Building FAISS index...", flush=True)
    index, _ = build_index(corpus, model)
    check_time(started, "embedding/index construction")
    save_index(index, corpus, CFG.cache_dir, _data_fingerprint(CFG.data_dir))
    elapsed = time.monotonic() - started
    print(f"FINAL COUNTS: documents={document_count}; sections={len(documents)}; passages={len(corpus)}", flush=True)
    print(f"REBUILD COMPLETE in {elapsed / 60:.2f} minutes; vectors={index.ntotal}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except TimeoutError as error:
        print(f"REBUILD TIMEBOX STOP: {error}", flush=True)
        sys.exit(2)
