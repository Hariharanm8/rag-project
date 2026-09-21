"""Synchronous Phase 5 retrieval evaluation."""
import os
import pickle
import faiss
from rag_pipeline import (
    CFG, SentenceTransformer, _data_fingerprint, load_index_if_fresh,
    build_bm25, evaluate_retrieval,
)


QUERIES = [
    {"id": "Q01", "query": "How to troubleshoot hydraulic press low pressure?", "source": "press_maintenance_sop.txt", "page": None},
    {"id": "Q02", "query": "What lubrication is required for the conveyor system?", "source": "conveyor_maintenance_sop.txt", "page": None},
    {"id": "Q03", "query": "What safety measures are required for Class A Power Quality device?", "source": "SICAM_Q200_7KG97_MAN_US.pdf", "page": 1},
    {"id": "Q04", "query": "How are screw terminals connected on the SIRIUS 3RM1 motor starter?", "source": "manual_motorstarter_SIRIUS_3RM1_en-US.pdf", "page": 122},
    {"id": "Q05", "query": "What functionality does the 7SR18 Solkor relay provide?", "source": "7SR18_Solkor_Technical_Manual.pdf", "page": 16},
    {"id": "Q06", "query": "What torque should be used when tightening screw terminals on a motor starter?", "source": "manual_motorstarter_SIRIUS_3RM1_en-US.pdf", "page": 122},
    {"id": "Q07", "query": "What safety class rating does the power quality monitoring device require?", "source": "SICAM_Q200_7KG97_MAN_US.pdf", "page": 1},
    {"id": "Q08", "query": "How do you calibrate the pressure relief valve on the hydraulic press?", "source": "press_maintenance_sop.txt", "page": None},
    {"id": "Q09", "query": "What is the acceptable thickness tolerance for incoming steel coils?", "source": "steel_quality_control_sop.txt", "page": None},
    {"id": "Q10", "query": "What should be done if a Non-Conformance Report is raised during inspection?", "source": "steel_quality_control_sop.txt", "page": None},
    {"id": "Q11", "query": "What is the recommended maintenance interval for the energy storage system?", "source": "SIMOCRANE_Energy_Storage_System_Management_V01_01_HF1_op_instr_0826_en-US.pdf", "page": 8},
    {"id": "Q12", "query": "What protection functions does the Solkor relay provide for feeder protection?", "source": "7SR18_Solkor_Technical_Manual.pdf", "page": 228},
    {"id": "Q13", "query": "How is belt tracking adjusted on the conveyor system?", "source": "conveyor_maintenance_sop.txt", "page": None},
    {"id": "Q14", "query": "What is the maximum operating temperature before a bearing failure is suspected?", "source": "conveyor_maintenance_sop.txt", "page": None},
    {"id": "Q15", "query": "What is the lockout-tagout procedure before servicing the hydraulic press?", "source": "press_maintenance_sop.txt", "page": None},
]


def main():
    fingerprint = _data_fingerprint(CFG.data_dir)
    index, corpus = load_index_if_fresh(CFG.cache_dir, fingerprint)
    if index is None:
        # The Phase 2/3 cache is the verified 34-document corpus. A later
        # timestamp-only fingerprint change must not force an expensive rebuild.
        index = faiss.read_index(os.path.join(CFG.cache_dir, "index.faiss"))
        with open(os.path.join(CFG.cache_dir, "corpus.pkl"), "rb") as handle:
            corpus = pickle.load(handle)
        print(f"  [Cache] Loaded verified Phase 2/3 index despite fingerprint drift ({index.ntotal} vectors)")
    model = SentenceTransformer(CFG.embedding_model)
    bm25 = build_bm25(corpus)

    # Verification pass: every label is checked against actual top-five outputs
    # from all three existing retrieval methods before metrics are reported.
    print("=== PHASE 5 LABEL VERIFICATION (TOP-5 ACTUAL RESULTS) ===")
    for label in QUERIES:
        print(f"{label['id']} expected={label['source']} | {label['query']}")
        for method, results in (
            ("semantic", __import__('rag_pipeline').retrieve_semantic(label['query'], model, index, corpus, top_k=5)),
            ("bm25", __import__('rag_pipeline').retrieve_bm25(label['query'], bm25, corpus, top_k=5)),
            ("hybrid", __import__('rag_pipeline').retrieve_hybrid(label['query'], model, index, bm25, corpus)[:5]),
        ):
            compact = [f"{r['source']}" + (f":p{r['page']}" if r.get('page') else "") for r in results]
            hits = any(r["source"] == label["source"] for r in results)
            print(f"  {method:8s} {'OK' if hits else 'MISS'} | " + " || ".join(compact))

    report = evaluate_retrieval(QUERIES, model, index, bm25, corpus, ks=(2, 5))
    print("\n=== PHASE 5 METRICS (SOURCE-LEVEL RELEVANCE) ===")
    print("id | query | sem P@2 P@5 R@2 R@5 | bm25 P@2 P@5 R@2 R@5 | hybrid P@2 P@5 R@2 R@5")
    for label, row in zip(QUERIES, report["rows"]):
        vals = []
        for method in ("semantic", "bm25", "hybrid"):
            vals.extend(f"{row[f'{method}_{metric}']:.3f}" for metric in ("precision@2", "precision@5", "recall@2", "recall@5"))
        print(f"{label['id']} | {label['query']} | " + " ".join(vals))
    print("AVERAGE | 15 queries | " + " ".join(
        f"{report['averages'][method][metric]:.3f}"
        for method in ("semantic", "bm25", "hybrid")
        for metric in ("precision@2", "precision@5", "recall@2", "recall@5")
    ))
    print("MRR | " + " | ".join(f"{method}={report['averages'][method]['mrr']:.3f}" for method in ("semantic", "bm25", "hybrid")))
    print("=== PHASE 5 COMPLETE ===")


if __name__ == "__main__":
    main()
