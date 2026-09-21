"""
generate_groq.py
=================
Groq-based generation for the RAG pipeline. Paste this file into your
project folder (same folder as rag_pipeline.py), or copy the function
below directly into rag_pipeline.py if you prefer one file.

Setup (one-time, already done if you followed the earlier steps):
  1. pip install groq
  2. Set the GROQ_API_KEY environment variable (setx GROQ_API_KEY "gsk_...")
  3. Close and reopen your terminal so the variable is available

This keeps your existing local generate_answer() function untouched --
this is a new, separate function you call instead of (or alongside) it.
"""

import os
import time
from groq import Groq

# Groq client reads GROQ_API_KEY from the environment automatically.
# We check it explicitly first so we get a clear error instead of a
# confusing one later.
_api_key = os.environ.get("GROQ_API_KEY")
if not _api_key:
    raise RuntimeError(
        "GROQ_API_KEY is not set. Run: setx GROQ_API_KEY \"gsk_your_key\" "
        "in PowerShell, then close and reopen your terminal."
    )

client = Groq(api_key=_api_key)

# Primary model: higher quality. Fallback: much higher daily request limit,
# useful if you're running many evaluation queries (Phase 5) and hit the
# primary model's daily cap.
PRIMARY_MODEL = "openai/gpt-oss-20b"      # good quality, what Codex already wired up
FALLBACK_MODEL = "llama-3.1-8b-instant"   # higher RPD if you hit rate limits


MIN_WORDS = 300
MAX_WORDS = 1000


def _build_prompt(query, retrieved_chunks):
    """Formats retrieved chunks into a grounded prompt. Mirrors the same
    grounding behavior your rag_pipeline.py version has: answer ONLY from
    the given context, cite source+page, refuse rather than guess if the
    context doesn't support an answer -- and now, elaborate thoroughly
    rather than answering tersely."""
    context_blocks = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        page = chunk.get("page")
        page_str = f", page {page}" if page else ""
        context_blocks.append(
            f"[Source {i}: {chunk['source']}{page_str}]\n{chunk['text']}"
        )
    context = "\n\n".join(context_blocks)

    system_prompt = (
        "You are a precise, thorough technical assistant answering questions using ONLY the "
        "provided document excerpts. Rules:\n"
        "1. Answer only using information explicitly present in the context below.\n"
        "2. If the context does not contain enough information to answer "
        "confidently, say so clearly instead of guessing or inferring.\n"
        "3. Always cite the specific source document (and page number if "
        "given) for each claim you make, in the format [source, p.X].\n"
        f"4. Write a thorough, well-elaborated answer between {MIN_WORDS} and "
        f"{MAX_WORDS} words -- explain the relevant procedure, values, and "
        "context in full technical detail. Do NOT give a one-line or "
        "one-word answer, and do not needlessly pad with repetition either; "
        "elaborate only on what the context actually supports."
    )

    user_prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    return system_prompt, user_prompt


def generate_answer_groq(query, retrieved_chunks, model=PRIMARY_MODEL, max_retries=1):
    """
    Generate a grounded, cited answer using the Groq API.

    query: the user's question (string)
    retrieved_chunks: list of dicts from retrieve_hybrid(), each with at
                       least 'text', 'source', and optionally 'page'
    model: which Groq model to use (see PRIMARY_MODEL / FALLBACK_MODEL above)
    max_retries: how many times to retry once on rate-limit (429) before
                 giving up and reporting clearly -- deliberately NOT an
                 aggressive retry loop.

    Returns: dict with 'answer' (string), 'model_used', and 'latency_seconds'
    """
    system_prompt, user_prompt = _build_prompt(query, retrieved_chunks)

    t0 = time.time()
    attempt = 0
    current_model = model

    while True:
        try:
            request_kwargs = dict(
                model=current_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,  # low temperature: favors grounded, factual answers
                max_tokens=1600,  # generous budget for a 300-1000 word answer
            )
            # gpt-oss reasoning models spend part of the token budget on
            # hidden reasoning before the visible answer -- keep it low so
            # the budget goes to the actual elaborated answer, not thinking.
            if "gpt-oss" in current_model:
                request_kwargs["reasoning_effort"] = "low"
                request_kwargs["max_completion_tokens"] = request_kwargs.pop("max_tokens")

            response = client.chat.completions.create(**request_kwargs)
            answer = response.choices[0].message.content
            latency = time.time() - t0
            return {
                "answer": answer,
                "model_used": current_model,
                "latency_seconds": round(latency, 2),
            }

        except Exception as e:
            error_str = str(e).lower()
            is_rate_limit = "429" in error_str or "rate_limit" in error_str

            if is_rate_limit and attempt < max_retries:
                # Back off once, then try the fallback model (which has a
                # much higher daily request cap) instead of hammering the
                # same rate-limited model again.
                print(f"  [Groq] Rate limited on {current_model}, waiting 5s "
                      f"then trying {FALLBACK_MODEL}...")
                time.sleep(5)
                current_model = FALLBACK_MODEL
                attempt += 1
                continue

            # Either not a rate-limit error, or we've already retried once --
            # report clearly instead of looping.
            latency = time.time() - t0
            return {
                "answer": None,
                "error": str(e),
                "model_used": current_model,
                "latency_seconds": round(latency, 2),
            }


# ---------------------------------------------------------------------
# QUICK TEST -- run this file directly to sanity-check the Groq connection
# with one simple question before wiring it into the full pipeline.
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("Testing Groq connection...")
    fake_chunks = [
        {
            "text": "The relief valve should trigger at 3000 PSI +/- 50 PSI.",
            "source": "sop_hydraulic_press.txt",
            "page": None,
        }
    ]
    result = generate_answer_groq(
        "At what pressure should the relief valve trigger?", fake_chunks
    )
    print(f"\nModel used: {result['model_used']}")
    print(f"Latency: {result['latency_seconds']}s")
    if result.get("answer"):
        print(f"\nAnswer:\n{result['answer']}")
    else:
        print(f"\nError: {result.get('error')}")
