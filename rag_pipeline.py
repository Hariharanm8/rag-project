"""
Enterprise Document RAG Intelligence Platform
=================================================
Architecture & Production Features:
  1. Multi-Format Document Ingestion (.pdf via pdfplumber + pypdf fallback, .txt)
  2. Automated Technical Noise Removal (strips headers, footers, pagination artefacts)
  3. Page & Section Metadata Tracking (enables exact page-level citations)
  4. Sentence-Boundary Aware Overlapping Chunker (preserves grammatical integrity)
  5. Persistent Index & Corpus Cache (SHA-256 fingerprinting for zero-cost restart)
  6. Dense Semantic Retrieval (FAISS vector index using SentenceTransformers)
  7. Sparse Keyword Retrieval (BM25Okapi rank scoring)
  8. Hybrid Retrieval Engine (Min-Max Normalised Fusion: 70% Dense + 30% Sparse)
  9. Cross-Encoder Re-Ranking (MS-MARCO MiniLM L6 precision re-ranking)
  10. Metadata / Document Filtering (scope search to specific document filenames)
  11. LLM Contextual Answer Generation (Groq Cloud inference, free tier —
      local Qwen/FLAN kept as fallback if GROQ_API_KEY is unavailable)
  12. Hallucination & Confidence Guardrail (automatic relevance & coverage check)
  13. Enterprise Interactive Shell (rich commands: filter, info, log, export, help)
  14. Structured JSONL Audit Logging (session query history & latency tracking)

Retrieval, embedding, reranking, and OCR run 100% locally/free. Generation
uses Groq's free-tier API by default (requires GROQ_API_KEY); falls back to
a local model automatically if the key or SDK is unavailable.
"""

import os
import sys
import glob
import re
import json
import time
import pickle
import hashlib
import datetime
import tempfile
import numpy as np
from dataclasses import dataclass, field
from sentence_transformers import SentenceTransformer
import faiss

# ---- Optional Dependecy Handling ----
try:
    from sentence_transformers.cross_encoder import CrossEncoder
    _CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CrossEncoder = None
    _CROSS_ENCODER_AVAILABLE = False

try:
    import pdfplumber
    _PDFPLUMBER_AVAILABLE = True
except ImportError:
    pdfplumber = None
    _PDFPLUMBER_AVAILABLE = False

try:
    from pypdf import PdfReader
    _PYPDF_AVAILABLE = True
except ImportError:
    PdfReader = None
    _PYPDF_AVAILABLE = False

try:
    from gradio_client import Client
    _OVISOCR_AVAILABLE = True
except ImportError:
    Client = None
    _OVISOCR_AVAILABLE = False

try:
    from rank_bm25 import BM25Okapi
    _BM25_AVAILABLE = True
except ImportError:
    BM25Okapi = None
    _BM25_AVAILABLE = False

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    import torch
    _TRANSFORMERS_LLM_AVAILABLE = True
except ImportError:
    AutoModelForCausalLM = None
    AutoTokenizer = None
    pipeline = None
    torch = None
    _TRANSFORMERS_LLM_AVAILABLE = False

try:
    from groq import Groq
    _GROQ_AVAILABLE = True
except ImportError:
    Groq = None
    _GROQ_AVAILABLE = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# =====================================================================
#  CENTRAL CONFIGURATION MANAGEMENT
# =====================================================================

@dataclass
class Config:
    # Chunking Strategy
    chunk_size: int    = 400    # Target max words per chunk
    chunk_overlap: int = 80     # Word overlap between contiguous chunks

    # Retrieval Tuning
    retrieval_pool: int    = 20   # Candidate chunks fetched prior to re-ranking
    final_top_k: int       = 4    # Top passages passed to generator / display
    semantic_weight: float = 0.7  # Dense vector weight in hybrid score
    bm25_weight: float     = 0.3  # Sparse BM25 weight in hybrid score
    confidence_threshold: float = 0.25 # Minimum score threshold for high confidence

    # Models Configuration
    embedding_model: str = "all-MiniLM-L6-v2"
    reranker_model: str  = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    generator_backend: str = "groq"
    generator_model: str = "openai/gpt-oss-20b"
    generator_reasoning_effort: str = "low"  # gpt-oss models: minimize hidden
                                              # reasoning tokens so the token
                                              # budget goes to the visible answer
    generator_max_new_tokens: int = 1600     # generous budget: covers a
                                              # 300-1000 word answer even after
                                              # reasoning-token overhead
    generator_min_words: int = 300
    generator_max_words: int = 1000
    generator_context_char_limit: int = 9_000
    use_reranker: bool   = True
    use_generator: bool  = True   # Set False for retrieval-only mode

    # Interface & Logging
    snippet_length: int = 300
    data_dir:  str = field(default_factory=lambda: os.path.join(SCRIPT_DIR, "data"))
    cache_dir: str = field(default_factory=lambda: os.path.join(SCRIPT_DIR, ".cache"))
    log_file:  str = field(default_factory=lambda: os.path.join(SCRIPT_DIR, "query_log.jsonl"))
    ocr_space: str = "ATH-MaaS/OvisOCR2"
    ocr_timeout_seconds: float = 90.0
    ocr_min_text_chars: int = 20
    # Set RAG_ENABLE_OCR_FALLBACK=0 for deterministic, local-only rebuilds.
    enable_ocr_fallback: bool = field(
        default_factory=lambda: os.environ.get("RAG_ENABLE_OCR_FALLBACK", "1") == "1"
    )


CFG = Config()
_INGESTION_CACHE_VERSION = (
    "pdfplumber-ovisocr2-fallback-v1"
    if CFG.enable_ocr_fallback
    else "pdfplumber-no-ocr-fallback-v1"
)
_ocr_client = None
_ocr_client_initialized = False


# =====================================================================
#  DOCUMENT INGESTION & PRE-PROCESSING
# =====================================================================

def _clean_page_text(text: str) -> str:
    """Strips repetitive headers, footers, bare page numbers, and PDF artefacts."""
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if re.fullmatch(r"\d+", s):                 # Bare page numbers
            continue
        if re.search(r"^E\d{5}-|^Doc(ument)?\s*Version|^Edition:\s*\d+", s, re.IGNORECASE):
            continue                                # Revision numbers/headers
        if len(s) < 5:                              # Fragmented artefacts
            continue
        cleaned.append(s)
    result = " ".join(cleaned)
    return re.sub(r" {2,}", " ", result).strip()


def _read_pdf_pdfplumber(filepath: str) -> list:
    """Extracts text page-by-page using pdfplumber."""
    pages = []
    try:
        with pdfplumber.open(filepath) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                raw = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
                cleaned = _clean_page_text(raw)
                if cleaned:
                    pages.append((i, cleaned))
    except Exception as e:
        print(f"  [WARN] pdfplumber error on {os.path.basename(filepath)}: {e}")
    return pages


def _get_ovis_ocr_client():
    """Creates the optional OvisOCR2 client once, with a bounded HTTP timeout."""
    global _ocr_client, _ocr_client_initialized
    if _ocr_client_initialized:
        return _ocr_client
    _ocr_client_initialized = True
    if not _OVISOCR_AVAILABLE:
        print("  [OCR] gradio_client is unavailable; skipping OCR fallback.")
        return None
    try:
        _ocr_client = Client(
            CFG.ocr_space,
            verbose=False,
            httpx_kwargs={"timeout": CFG.ocr_timeout_seconds},
        )
    except Exception as error:
        print(f"  [OCR] OvisOCR2 client unavailable: {error}")
    return _ocr_client


def _normalise_ocr_result(result) -> str:
    """Extracts text from the string, list, or dictionary returned by Gradio."""
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        preferred_keys = ("text", "output_text", "result", "markdown", "content")
        for key in preferred_keys:
            if key in result:
                text = _normalise_ocr_result(result[key])
                if text:
                    return text
        for value in result.values():
            text = _normalise_ocr_result(value)
            if text:
                return text
    if isinstance(result, (list, tuple)):
        for value in result:
            text = _normalise_ocr_result(value)
            if text:
                return text
    return ""


def _ocr_pdfplumber_page(page, page_number: int, page_count: int) -> str:
    """Renders one PDF page and sends it once to the verified OvisOCR2 endpoint."""
    client = _get_ovis_ocr_client()
    if client is None:
        return ""

    image_path = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as image_file:
            image_path = image_file.name
        page.to_image(resolution=150).save(image_path, format="PNG")
    except Exception as error:
        print(f"    [OCR] page render failed: {error!r}")
        return ""

    try:
        result = client.predict(
            image_path=image_path,
            page_index=page_number - 1,
            page_count=page_count,
            api_name="/run_ocr",
        )
        return _clean_page_text(_normalise_ocr_result(result))
    except Exception as error:
        print(f"    [OCR] request failed: {error!r}")
        return ""
    finally:
        if image_path and os.path.exists(image_path):
            os.remove(image_path)


def _read_pdf_pdfplumber_with_ocr(filepath: str) -> tuple[list, dict]:
    """Uses pdfplumber first and OvisOCR2 only for individual insufficient pages."""
    pages = []
    stats = {
        "pdfplumber_pages": 0,
        "ocr_attempted": 0,
        "ocr_recovered": 0,
        "ocr_skipped": 0,
    }
    filename = os.path.basename(filepath)
    try:
        with pdfplumber.open(filepath) as pdf:
            page_count = len(pdf.pages)
            for page_number, page in enumerate(pdf.pages, start=1):
                raw = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
                cleaned = _clean_page_text(raw)
                if len(cleaned) >= CFG.ocr_min_text_chars:
                    pages.append((page_number, cleaned))
                    stats["pdfplumber_pages"] += 1
                    continue

                if not CFG.enable_ocr_fallback:
                    stats["ocr_skipped"] += 1
                    print(
                        f"    [OCR disabled] {filename} p.{page_number}: pdfplumber returned "
                        f"{len(cleaned)} usable chars; page omitted."
                    )
                    continue

                stats["ocr_attempted"] += 1
                print(
                    f"    [OCR] {filename} p.{page_number}: pdfplumber returned "
                    f"{len(cleaned)} usable chars; requesting OvisOCR2."
                )
                ocr_text = _ocr_pdfplumber_page(page, page_number, page_count)
                if ocr_text:
                    pages.append((page_number, ocr_text))
                    stats["ocr_recovered"] += 1
                    print(f"    [OCR] {filename} p.{page_number}: recovered {len(ocr_text):,} chars.")
                else:
                    print(f"    [OCR] {filename} p.{page_number}: no usable OCR text returned.")
    except Exception as error:
        print(f"  [WARN] pdfplumber error on {filename}: {error}")
    return pages, stats


def _read_pdf_pypdf(filepath: str) -> list:
    """Fallback PDF extraction using pypdf."""
    if not _PYPDF_AVAILABLE:
        return []
    pages = []
    try:
        reader = PdfReader(filepath)
        for i, page in enumerate(reader.pages, start=1):
            raw = page.extract_text() or ""
            cleaned = _clean_page_text(raw)
            if cleaned:
                pages.append((i, cleaned))
    except Exception as e:
        print(f"  [WARN] pypdf error on {os.path.basename(filepath)}: {e}")
    return pages


def load_documents(folder_path: str, progress_interval: int | None = None) -> list:
    """Loads all text and PDF files from data folder into structured documents."""
    documents = []

    file_paths = sorted(
        glob.glob(os.path.join(folder_path, "*.txt"))
        + glob.glob(os.path.join(folder_path, "*.pdf"))
    )
    total_files = len(file_paths)

    for file_number, fp in enumerate(file_paths, start=1):
        fname = os.path.basename(fp)
        if progress_interval and (file_number == 1 or file_number % progress_interval == 0 or file_number == total_files):
            print(f"  [Progress] ingesting file {file_number}/{total_files}: {fname}")

        if fp.lower().endswith(".txt"):
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                if text:
                    documents.append({"filename": fname, "page": None, "text": text})
                    print(f"  [Txt]  {fname} ({len(text):,} chars)")
                else:
                    print(f"  [Skip] {fname} (empty file)")
            except Exception as e:
                print(f"  [WARN] {fname}: {e}")
            continue

        # --- PDF Files ---
        try:
            if _PDFPLUMBER_AVAILABLE:
                pages, ocr_stats = _read_pdf_pdfplumber_with_ocr(fp)
                engine = "pdfplumber"
            else:
                pages, engine = _read_pdf_pypdf(fp), "pypdf"
                ocr_stats = {"pdfplumber_pages": 0, "ocr_attempted": 0, "ocr_recovered": 0, "ocr_skipped": 0}

            if pages:
                chars = sum(len(t) for _, t in pages)
                for pg, text in pages:
                    documents.append({"filename": fname, "page": pg, "text": text})
                if ocr_stats["ocr_attempted"]:
                    print(
                        f"  [Pdf]  {fname} ({len(pages)} pages, {chars:,} chars) via {engine}; "
                        f"fast path {ocr_stats['pdfplumber_pages']}, OCR {ocr_stats['ocr_recovered']}/"
                        f"{ocr_stats['ocr_attempted']} pages recovered"
                    )
                elif ocr_stats["ocr_skipped"]:
                    print(
                        f"  [Pdf]  {fname} ({len(pages)} pages, {chars:,} chars) via {engine}; "
                        f"fast path {ocr_stats['pdfplumber_pages']}, OCR disabled; "
                        f"{ocr_stats['ocr_skipped']} insufficient page(s) omitted"
                    )
                else:
                    print(f"  [Pdf]  {fname} ({len(pages)} pages, {chars:,} chars) via {engine}; fast path only")
            else:
                print(f"  [Skip] {fname} (no text extracted)")
        except Exception as e:
            print(f"  [WARN] {fname}: {e}")

    return documents


# =====================================================================
#  SENTENCE-BOUNDED CHUNKING ENGINE
# =====================================================================

def _split_sentences(text: str) -> list:
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> list:
    """Sentence-boundary preserving chunker."""
    chunk_size = chunk_size or CFG.chunk_size
    overlap    = overlap    or CFG.chunk_overlap

    sentences = _split_sentences(text)
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


def build_chunk_corpus(documents: list) -> list:
    """Transforms raw document sections into indexed chunk dicts."""
    corpus = []
    for doc in documents:
        for chunk in chunk_text(doc["text"]):
            corpus.append({
                "text": chunk,
                "source": doc["filename"],
                "page": doc["page"]
            })
    return corpus


# =====================================================================
#  PERSISTENCE & CACHING LAYER
# =====================================================================

def _data_fingerprint(folder_path: str) -> str:
    """SHA-256 checksum of document data and ingestion behavior."""
    h = hashlib.sha256()
    h.update(_INGESTION_CACHE_VERSION.encode())
    for fp in sorted(glob.glob(os.path.join(folder_path, "*.*"))):
        h.update(fp.encode())
        h.update(str(os.path.getmtime(fp)).encode())
    return h.hexdigest()


def _cache_paths(cache_dir: str):
    os.makedirs(cache_dir, exist_ok=True)
    return (
        os.path.join(cache_dir, "index.faiss"),
        os.path.join(cache_dir, "corpus.pkl"),
        os.path.join(cache_dir, "fingerprint.txt"),
    )


def save_index(index, corpus: list, cache_dir: str, fingerprint: str):
    idx_p, corpus_p, fp_p = _cache_paths(cache_dir)
    faiss.write_index(index, idx_p)
    with open(corpus_p, "wb") as f:
        pickle.dump(corpus, f)
    with open(fp_p, "w") as f:
        f.write(fingerprint)
    print(f"  [Cache] Saved index and corpus to {cache_dir}")


def load_index_if_fresh(cache_dir: str, fingerprint: str):
    idx_p, corpus_p, fp_p = _cache_paths(cache_dir)
    if not all(os.path.exists(p) for p in (idx_p, corpus_p, fp_p)):
        return None, None
    with open(fp_p) as f:
        cached_fp = f.read().strip()
    if cached_fp != fingerprint:
        return None, None
    index = faiss.read_index(idx_p)
    with open(corpus_p, "rb") as f:
        corpus = pickle.load(f)
    print(f"  [Cache] Fast-loaded pre-built index ({index.ntotal} vectors)")
    return index, corpus


# =====================================================================
#  DENSE (FAISS) & SPARSE (BM25) RETRIEVAL ENGINES
# =====================================================================

def build_index(corpus: list, model):
    texts = [c["text"] for c in corpus]
    print(f"  Encoding {len(texts)} chunks with {model.get_sentence_embedding_dimension()}D embeddings...")
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=64,
    )
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype(np.float32))
    return index, embeddings


def retrieve_semantic(query: str, model, index, corpus: list, doc_filter: str = None, top_k: int = 20) -> list:
    qvec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    # Search deeper to account for potential doc filtering
    search_k = min(top_k * 4, len(corpus))
    scores, idxs = index.search(qvec.astype(np.float32), search_k)

    results = []
    for i in range(len(idxs[0])):
        idx = int(idxs[0][i])
        c = corpus[idx]
        if doc_filter and doc_filter.lower() not in c["source"].lower():
            continue
        results.append({
            "score": float(scores[0][i]),
            "idx": idx,
            "source": c["source"],
            "page": c["page"],
            "text": c["text"]
        })
        if len(results) >= top_k:
            break
    return results


def build_bm25(corpus: list):
    if not _BM25_AVAILABLE:
        print("  [WARN] rank-bm25 not installed. Sparse retrieval disabled.")
        return None
    tokenized = [c["text"].lower().split() for c in corpus]
    return BM25Okapi(tokenized)


def retrieve_bm25(query: str, bm25, corpus: list, doc_filter: str = None, top_k: int = 20) -> list:
    if bm25 is None:
        return []
    scores = bm25.get_scores(query.lower().split())
    top_idx = np.argsort(scores)[::-1]

    results = []
    for idx in top_idx:
        c = corpus[idx]
        if doc_filter and doc_filter.lower() not in c["source"].lower():
            continue
        results.append({
            "score": float(scores[idx]),
            "idx": int(idx),
            "source": c["source"],
            "page": c["page"],
            "text": c["text"]
        })
        if len(results) >= top_k:
            break
    return results


def retrieve_hybrid(query: str, model, index, bm25, corpus: list, doc_filter: str = None) -> list:
    """Combines Dense (Vector) and Sparse (BM25) results using Min-Max Normalized Fusion."""
    pool = min(CFG.retrieval_pool, len(corpus))
    sem_res  = retrieve_semantic(query, model, index, corpus, doc_filter=doc_filter, top_k=pool)
    bm25_res = retrieve_bm25(query, bm25, corpus, doc_filter=doc_filter, top_k=pool)

    sem_map  = {r["idx"]: r["score"] for r in sem_res}
    bm25_map = {r["idx"]: r["score"] for r in bm25_res}

    def _minmax(d):
        if not d: return {}
        lo, hi = min(d.values()), max(d.values())
        if hi == lo: return {k: 1.0 for k in d}
        return {k: (v - lo) / (hi - lo) for k, v in d.items()}

    sn, bn  = _minmax(sem_map), _minmax(bm25_map)
    all_idx = set(sn) | set(bn)

    combined = {
        i: CFG.semantic_weight * sn.get(i, 0.0) + CFG.bm25_weight * bn.get(i, 0.0)
        for i in all_idx
    }

    top_idx = sorted(combined, key=combined.get, reverse=True)[:CFG.retrieval_pool]
    return [
        {
            "score": combined[i],
            "idx": i,
            "source": corpus[i]["source"],
            "page": corpus[i]["page"],
            "text": corpus[i]["text"]
        }
        for i in top_idx
    ]


def evaluate_retrieval(query_set: list, model, index, bm25, corpus: list, ks=(2, 5)) -> dict:
    """Evaluate dense, BM25, and hybrid retrieval against labeled source documents.

    Each query label must contain ``query`` and ``source``; ``page`` is optional
    metadata for reporting. Relevance is source-level: a returned passage is
    relevant when its ``source`` equals the label's source. Recall is therefore
    binary for this one-document ground truth (whether at least one relevant
    passage appears in top-k), while precision is the fraction of top-k
    passages from that source. Existing retrieval functions are called directly.
    """
    methods = {
        "semantic": lambda q, k: retrieve_semantic(q, model, index, corpus, top_k=k),
        "bm25": lambda q, k: retrieve_bm25(q, bm25, corpus, top_k=k),
        "hybrid": lambda q, k: retrieve_hybrid(q, model, index, bm25, corpus)[:k],
    }
    rows, aggregate = [], {m: {f"precision@{k}": [] for k in ks} for m in methods}
    for label in query_set:
        row = {"query": label["query"], "source": label["source"], "page": label.get("page")}
        for method, retrieve in methods.items():
            ranked = retrieve(label["query"], max(ks))
            first_rank = next((i + 1 for i, item in enumerate(ranked)
                               if item.get("source") == label["source"]), None)
            row[f"{method}_mrr"] = 1.0 / first_rank if first_rank else 0.0
            for k in ks:
                top = ranked[:k]
                hits = sum(item.get("source") == label["source"] for item in top)
                row[f"{method}_precision@{k}"] = hits / k
                row[f"{method}_recall@{k}"] = 1.0 if hits else 0.0
                aggregate[method].setdefault(f"precision@{k}", []).append(hits / k)
                aggregate[method].setdefault(f"recall@{k}", []).append(1.0 if hits else 0.0)
            aggregate[method].setdefault("mrr", []).append(row[f"{method}_mrr"])
        rows.append(row)
    averages = {m: {metric: float(np.mean(values)) for metric, values in metrics.items()}
                for m, metrics in aggregate.items()}
    return {"rows": rows, "averages": averages, "ks": tuple(ks)}


# =====================================================================
#  CROSS-ENCODER RE-RANKER
# =====================================================================

def load_reranker():
    if not CFG.use_reranker or not _CROSS_ENCODER_AVAILABLE:
        return None
    print(f"  Loading Cross-Encoder Re-ranker ({CFG.reranker_model})...")
    try:
        return CrossEncoder(CFG.reranker_model)
    except Exception as e:
        print(f"  [WARN] CrossEncoder load failed: {e}")
        return None


def rerank(query: str, candidates: list, reranker) -> list:
    if not reranker or not candidates:
        return candidates
    pairs = [(query, c["text"]) for c in candidates]
    ce_scores = reranker.predict(pairs)
    for c, s in zip(candidates, ce_scores):
        c["rerank_score"] = float(s)
    return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)


# =====================================================================
#  CITATION-GROUNDED ANSWER SYNTHESIS & GENERATION ENGINE
# =====================================================================

def load_groq_generator():
    """Creates a Groq client using GROQ_API_KEY without exposing the secret."""
    if not CFG.use_generator:
        return None
    if not _GROQ_AVAILABLE:
        print("  [WARN] Groq SDK is not installed. Operating in Extractive Mode.")
        return None
    if not os.environ.get("GROQ_API_KEY"):
        print("  [WARN] GROQ_API_KEY is not set. Operating in Extractive Mode.")
        return None
    try:
        return {"client": Groq(), "model_name": CFG.generator_model, "backend": "groq"}
    except Exception as error:
        print(f"  [WARN] Groq client initialization skipped ({error}). Operating in Extractive Mode.")
        return None

def load_qwen_generator():
    """Loads the configured Qwen instruct model on CPU with memory-conscious weights."""
    if not CFG.use_generator or not _TRANSFORMERS_LLM_AVAILABLE:
        return None
    print(f"  Loading Local Qwen Generator ({CFG.generator_model})...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(CFG.generator_model)
        model = AutoModelForCausalLM.from_pretrained(
            CFG.generator_model,
            torch_dtype=torch.bfloat16,
        )
        model.eval()
        return {"model": model, "tokenizer": tokenizer, "model_name": CFG.generator_model}
    except Exception as error:
        print(f"  [WARN] Qwen generator initialization skipped ({error}). Operating in Extractive Mode.")
        return None


def load_generator():
    """Loads the configured backend while retaining the legacy local-model paths."""
    if CFG.generator_backend == "groq":
        return load_groq_generator()
    if CFG.generator_model.startswith("Qwen/"):
        return load_qwen_generator()
    if not CFG.use_generator or not _TRANSFORMERS_LLM_AVAILABLE:
        return None
    print(f"  Loading Local Generator ({CFG.generator_model})...")
    try:
        return pipeline(
            "text2text-generation",
            model=CFG.generator_model,
            max_new_tokens=CFG.generator_max_new_tokens,
            do_sample=False,
        )
    except Exception as error:
        print(f"  [WARN] Local generator initialization skipped ({error}). Operating in Extractive Mode.")
        return None


def _build_grounded_context(retrieved_chunks: list) -> tuple[str, list]:
    """Formats retrieved passages with source-and-page citations for generation."""
    context_blocks = []
    citations = []
    remaining_chars = CFG.generator_context_char_limit
    for position, chunk in enumerate(retrieved_chunks[:3], start=1):
        citation = f"[{chunk['source']}" + (f", p.{chunk['page']}]" if chunk.get("page") else "]")
        text = chunk["text"].strip()
        if not text or remaining_chars <= 0:
            continue
        text = text[:remaining_chars]
        context_blocks.append(f"Passage {position} {citation}:\n{text}")
        citations.append(citation)
        remaining_chars -= len(text)
    return "\n\n".join(context_blocks), list(dict.fromkeys(citations))


def generate_answer(query: str, retrieved_chunks: list, generator=None) -> str:
    """Returns a citation-grounded answer using only the supplied retrieved context."""
    if not retrieved_chunks:
        return "I cannot answer because the retrieved context contains no relevant passages."

    context, citations = _build_grounded_context(retrieved_chunks)
    if not context:
        return "I cannot answer because the retrieved context contains no usable passages."

    prompt = (
        "Answer only from the supplied enterprise-document context. Do not add outside "
        "knowledge or assumptions. If the context does not answer the question, say so. "
        "Cite each factual statement using the exact source-and-page citation shown in its "
        "passage.\n\n"
        f"Write a thorough, well-elaborated answer between {CFG.generator_min_words} and "
        f"{CFG.generator_max_words} words. Explain the relevant procedure, values, and "
        "reasoning found in the context in full detail rather than a short summary -- expand "
        "on every relevant point the context supports, in clear technical prose, while staying "
        "strictly grounded in the provided passages.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}"
    )
    answer = ""
    try:
        if isinstance(generator, dict) and generator.get("backend") == "groq":
            completion_kwargs = dict(
                model=generator["model_name"],
                messages=[
                    {"role": "system", "content": "You are a precise, thorough technical RAG assistant. Follow the user's grounding, citation, and length instructions exactly -- do not give one-line or one-word answers."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_completion_tokens=CFG.generator_max_new_tokens,
            )
            # gpt-oss reasoning models spend part of the token budget on
            # hidden reasoning before the visible answer -- keep that low so
            # the budget goes to the actual elaborated answer.
            if "gpt-oss" in generator["model_name"]:
                completion_kwargs["reasoning_effort"] = CFG.generator_reasoning_effort
            completion = generator["client"].chat.completions.create(**completion_kwargs)
            answer = (completion.choices[0].message.content or "").strip()
        elif isinstance(generator, dict) and "model" in generator and "tokenizer" in generator:
            tokenizer = generator["tokenizer"]
            model = generator["model"]
            messages = [
                {"role": "system", "content": "You are a precise technical RAG assistant."},
                {"role": "user", "content": prompt},
            ]
            rendered_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(rendered_prompt, return_tensors="pt")
            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=CFG.generator_max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
            answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        elif generator:
            result = generator(f"{prompt}\nAnswer:")
            answer = result[0]["generated_text"].strip()
    except Exception as error:
        print(f"  [WARN] Generation failed: {error}")

    if not answer:
        best_citation = citations[0] if citations else "[source unavailable]"
        answer = f"The retrieved context states: {retrieved_chunks[0]['text'][:350].strip()} {best_citation}"

    inference_markers = ("can be inferred", "would include", "typically found", "outside the provided")
    if any(marker in answer.lower() for marker in inference_markers):
        answer = (
            "The retrieved passages do not explicitly state enough detail to answer this "
            "question without making an unsupported inference."
        )

    if citations and not any(citation in answer for citation in citations):
        answer = f"{answer}\n\nSources: {', '.join(citations)}"
    return answer


def synthesize_answer(query: str, top_chunks: list, generator=None) -> dict:
    """Generates grounded answer with explicit citations and confidence guardrail."""
    if not top_chunks:
        return {
            "answer": "No relevant documents found matching the query.",
            "confidence": "Low",
            "citations": []
        }

    _, citations = _build_grounded_context(top_chunks)

    # Assess overall confidence score
    top_score = top_chunks[0].get("rerank_score", top_chunks[0].get("score", 0.0))
    if top_score > 0.6:
        confidence = "High"
    elif top_score > 0.35:
        confidence = "Medium"
    else:
        confidence = "Low (Informational match)"

    # Generate answer via local LLM if loaded
    if generator:
        return {
            "answer": generate_answer(query, top_chunks, generator=generator),
            "confidence": confidence,
            "citations": citations,
        }

    # Extractive Fallback Synthesis
    best_chunk = top_chunks[0]
    best_cite = citations[0]
    extract_snippet = best_chunk['text'][:350].strip()
    
    synthesized_answer = (
        f"Based on enterprise documentation {best_cite}:\n"
        f"\"{extract_snippet}...\"\n\n"
        f"Key Reference Sources: {', '.join(list(dict.fromkeys(citations)))}"
    )

    return {
        "answer": synthesized_answer,
        "confidence": confidence,
        "citations": citations
    }


# =====================================================================
#  SESSION AUDIT LOGGING
# =====================================================================

def log_query(query: str, results: list, synthesis: dict, latency_ms: float, log_file: str):
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "query": query,
        "latency_ms": round(latency_ms, 2),
        "confidence": synthesis.get("confidence", "N/A"),
        "answer_summary": synthesis.get("answer", "")[:150],
        "top_sources": [r["source"] for r in results[:3]],
        "top_passages": [
            {
                "source": r["source"],
                "page": r["page"],
                "score": round(r.get("rerank_score", r["score"]), 4),
                "text_snippet": r["text"][:180]
            }
            for r in results[:CFG.final_top_k]
        ]
    }
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        print(f"  [WARN] Audit log error: {e}")


# =====================================================================
#  CLI DISPLAY & INTERACTIVE SHELL
# =====================================================================

LINE  = "─" * 74
DOUBLE_LINE = "═" * 74

def print_banner(n_docs: int, n_chunks: int, reranker_on: bool, gen_on: bool):
    print(f"\n{DOUBLE_LINE}")
    print("  ENTERPRISE DOCUMENT INTELLIGENCE PLATFORM  [RAG Pipeline Engine]")
    print(DOUBLE_LINE)
    print(f"  Indexed Corpus   : {n_docs} Documents | {n_chunks} Semantic Passages")
    print(f"  Retrieval Engine : Hybrid (Dense 70% FAISS + Sparse 30% BM25)")
    print(f"  Re-Ranker Engine : {'ENABLED (' + CFG.reranker_model + ')' if reranker_on else 'DISABLED'}")
    print(f"  Generator LLM    : {'ENABLED (' + CFG.generator_model + ')' if gen_on else 'EXTRACTIVE MODE'}")
    print(f"  Audit Log Path   : {CFG.log_file}")
    print(LINE)
    print("  Enter your question below. Available Commands:")
    print("    'filter <doc>'  - Restrict search to specific document (e.g. filter press)")
    print("    'clearfilter'   - Clear document search scope")
    print("    'info'          - Show system statistics and index details")
    print("    'log'           - Show audit log path")
    print("    'exit' / 'quit' - Terminate interactive session")
    print(f"{DOUBLE_LINE}\n")


def display_rag_response(query: str, synthesis: dict, candidates: list, latency_ms: float, active_filter: str = None):
    print(f"\n{DOUBLE_LINE}")
    print(f"  QUESTION: {query}")
    if active_filter:
        print(f"  [Scope Filter Active: '{active_filter}']")
    print(DOUBLE_LINE)

    # Generated Answer
    print(f"\n  ► GENERATED ANSWER  (Confidence: {synthesis['confidence']} | Latency: {latency_ms:.1f}ms):")
    for line in synthesis["answer"].split("\n"):
        print(f"    {line}")

    # Retrieved Context Evidence
    print(f"\n{LINE}")
    print(f"  ► TOP RETRIEVED EVIDENCE PASSAGES ({min(len(candidates), CFG.final_top_k)} of {len(candidates)} candidates):")
    print(LINE)

    for i, r in enumerate(candidates[:CFG.final_top_k], start=1):
        score_val = r.get("rerank_score", r["score"])
        cite_str = f"{r['source']}" + (f" (Page {r['page']})" if r["page"] else "")
        snippet = r["text"][:CFG.snippet_length].replace("\n", " ") + "..."

        print(f"\n  [{i}] {cite_str}")
        print(f"      Relevance Score : {score_val:.4f}")
        print(f"      Excerpt         : \"{snippet}\"")

    print(f"\n{DOUBLE_LINE}\n")


def run_simple_interactive(model, index, bm25, generator, corpus: list):
    """Runs a minimal hybrid-retrieval and cited-generation prompt loop."""
    print("Simple RAG mode. Type 'exit' to quit.")
    while True:
        try:
            query = input("Ask a question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting simple RAG mode.")
            break

        if query.lower() == "exit":
            print("Exiting simple RAG mode.")
            break
        if not query:
            continue

        retrieved_chunks = retrieve_hybrid(query, model, index, bm25, corpus)
        answer = generate_answer(query, retrieved_chunks, generator=generator)
        print(f"\n{answer}\n")


def run_interactive(model, index, bm25, reranker, generator, corpus: list):
    unique_docs = len({c["source"] for c in corpus})
    print_banner(unique_docs, len(corpus), reranker is not None, generator is not None)

    active_filter = None
    query_count = 0

    while True:
        try:
            prompt_str = f"  RAG-Query [{active_filter}] > " if active_filter else "  RAG-Query > "
            raw = input(prompt_str).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  [Session Terminated]")
            break

        if not raw:
            continue

        cmd_lower = raw.lower()
        if cmd_lower in ("exit", "quit", "q"):
            print("  Exiting Enterprise Document Intelligence. Goodbye!")
            break

        if cmd_lower.startswith("filter "):
            active_filter = raw[7:].strip()
            print(f"  [Scope Filter Set]: Matches containing '{active_filter}'\n")
            continue

        if cmd_lower in ("clearfilter", "clear"):
            active_filter = None
            print("  [Scope Filter Cleared]: Searching all enterprise documents.\n")
            continue

        if cmd_lower == "info":
            print(f"\n  --- System Statistics ---")
            print(f"  FAISS Vectors : {index.ntotal}")
            print(f"  Corpus Chunks : {len(corpus)}")
            print(f"  Unique Files  : {unique_docs}")
            print(f"  Active Filter : {active_filter or 'None'}")
            print(f"  Log Location  : {CFG.log_file}\n")
            continue

        if cmd_lower == "log":
            print(f"\n  Audit Log Location: {CFG.log_file}\n")
            continue

        # Execute Pipeline
        t0 = time.time()

        # 1. Hybrid Search (Dense + Sparse)
        candidates = retrieve_hybrid(raw, model, index, bm25, corpus, doc_filter=active_filter)

        # 2. Cross-Encoder Re-Ranking
        if reranker and candidates:
            candidates = rerank(raw, candidates, reranker)

        # 3. Answer Generation & Synthesis
        synthesis = synthesize_answer(raw, candidates, generator=generator)

        latency_ms = (time.time() - t0) * 1000.0

        # 4. Display Results
        display_rag_response(raw, synthesis, candidates, latency_ms, active_filter=active_filter)

        # 5. Session Audit Log
        log_query(raw, candidates, synthesis, latency_ms, CFG.log_file)
        query_count += 1

    if query_count:
        print(f"  Session Completed: {query_count} queries recorded in {CFG.log_file}")


# =====================================================================
#  APPLICATION PIPELINE INITIALIZER
# =====================================================================

if __name__ == "__main__":
    print(DOUBLE_LINE)
    print("  INITIALIZING ENTERPRISE DOCUMENT INTELLIGENCE SYSTEM")
    print(DOUBLE_LINE)

    # 1. Validate data directory
    if not os.path.isdir(CFG.data_dir):
        print(f"[ERROR] Data folder missing at: {CFG.data_dir}")
        sys.exit(1)

    # 2. Check fingerprint for cached index
    fingerprint = _data_fingerprint(CFG.data_dir)
    print("\n[1/5] Checking Index Cache...")
    index, corpus = load_index_if_fresh(CFG.cache_dir, fingerprint)

    if index is None:
        print("\n[2/5] Ingesting Enterprise Documents...")
        documents = load_documents(CFG.data_dir)
        if not documents:
            print("[ERROR] No valid documents found in data/. Please add .pdf or .txt files.")
            sys.exit(1)
        print(f"      => {len(documents)} sections loaded across {len({d['filename'] for d in documents})} file(s)")

        print("\n[3/5] Chunking Documents (Sentence-Bounded)...")
        corpus = build_chunk_corpus(documents)
        print(f"      => Created {len(corpus)} search passages")

        print(f"\n[4/5] Loading Embedding Model ({CFG.embedding_model})...")
        model = SentenceTransformer(CFG.embedding_model)

        print("\n[5/5] Building FAISS Vector Index...")
        index, _ = build_index(corpus, model)
        save_index(index, corpus, CFG.cache_dir, fingerprint)
    else:
        print(f"\n[2/5] Loading Embedding Model ({CFG.embedding_model})...")
        model = SentenceTransformer(CFG.embedding_model)
        print("  [Cache] Loaded FAISS Vector Index and Chunk Corpus.")

    # Sparse BM25 Index
    print("\nBuilding BM25 Sparse Index...")
    bm25 = build_bm25(corpus)

    # Cross-Encoder Re-ranker
    print("\nLoading Cross-Encoder Re-ranker...")
    reranker = load_reranker()

    # Local LLM Generator
    print("\nLoading Generator Synthesis Engine...")
    generator = load_generator()

    # Launch Interactive Shell
    if "--simple" in sys.argv:
        run_simple_interactive(model, index, bm25, generator, corpus)
    else:
        run_interactive(model, index, bm25, reranker, generator, corpus)
