# AGENTS.md

You are continuing an existing project. Before doing anything else, and at the
start of EVERY new session:

1. Read `PROJECT_PLAN.md` in full, especially **Section 2 (Current Status
   Snapshot)** and **Section 4 (Phase Plan)**.
2. Identify the first phase in Section 4 that is not marked `DONE`.
3. Work ONLY on that phase's unchecked tasks. Do not skip ahead to later
   phases, and do not redo work already marked done — quickly verify it still
   works, then move forward from there.
4. Follow the principles in **Section 3 (Key Decisions & Principles)** exactly:
   - Free/local tools only (sentence-transformers, FAISS, a small local LLM) —
     no paid APIs.
   - Report real, honestly-measured results. Never invent or simulate numbers.
   - Don't overwrite existing working functions in `rag_pipeline.py` — extend
     it (e.g. add `hybrid_retrieve()` alongside `retrieve()` and
     `keyword_search()`, keep both working).
   - Every claim or statistic taken from a paper must be citable.
5. Before you finish a task, a phase, or before you're likely to run out of
   context/quota: **append a new dated entry to Section 5 (Session Log) in
   PROJECT_PLAN.md** describing exactly what you did, which files changed,
   and what remains. Then update the checkboxes in Section 4. Do this
   proactively — assume you may be cut off at any time, don't wait until the
   very end of a long task to log progress.
6. If you make a design decision not already covered in Section 3, add it as
   a new bullet there (not just in the log), so future sessions inherit it
   automatically.
7. At the start of your first reply in a session, briefly state: which phase
   you're starting on, and your plan for this session — before writing any
   code.

## Specific known issue to fix first

The previous session (in a different tool, Antigravity) burned most of its
quota stuck in retry loops trying to run background scripts and read their
output on Windows, plus PowerShell Unicode/UTF-8 encoding conversions for the
evidence `.txt` files. It's not confirmed whether the following files
actually contain complete, real output:
- `phase1_execution_evidence.txt`
- `phase2_execution_evidence.txt` / `phase2_evidence_34docs.txt`
- `phase3_execution_evidence.txt`

**Before doing anything else in Phase 1-3 territory:** open these files
directly and check if they contain real, complete query output (scores,
sources, snippets) or if they're empty/truncated/garbled. Report what you
find. Do NOT re-run the whole pipeline speculatively — check first, only
re-run the specific script whose evidence file is broken.

**When you do need to run a script, run it plainly and synchronously** —
`python script.py > out.txt 2>&1` followed immediately by reading the file —
rather than backgrounding it and polling/waiting in a loop. The previous
session's thrashing was caused by exactly that pattern.

Project files you should expect to find in this workspace:
- `PROJECT_PLAN.md` — full project state, phase plan, and session log (the
  source of truth — read it, don't just skim this file)
- `rag_pipeline.py` — main pipeline: ingestion, chunking, embedding, FAISS
  indexing, semantic retrieval, keyword baseline
- `demo_tfidf_sandbox.py` — a one-off verification script, not part of the
  real submission; ignore unless explicitly asked about it
- `data/` — 34 files: sample SOP/manual `.txt` documents plus real enterprise
  PDF manuals (Siemens SICAM Q200, SIRIUS 3RM1, SIMOCRANE, 7SR18 Solkor relay)
- `README.md` — setup/run instructions
- `Review2_Hariharan_22MIA1183.pptx` — the Review 2 slide deck (reference
  only, not something to regenerate unless asked)
