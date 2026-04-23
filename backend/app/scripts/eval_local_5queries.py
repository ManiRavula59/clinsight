import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

"""
eval_local_5queries.py
======================
Runs 5 queries ONE AT A TIME through the FULL Clinsight pipeline.
No batching, no parallelism — safe for Mac.

Each query goes through:
  1. Clinical Parser (NER + Negation)
  2. Knowledge Graph Structuring
  3. FAISS Dense Search
  4. FTS5 Lexical Search
  5. RRF Fusion
  6. ColBERT Reranking
  7. Cross-Encoder Reranking

Computes NDCG@10, Recall@10, MAP@10 per query, then averages.
"""

import os, sys, json, math, time, sqlite3, random

# Fix imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from typing import Dict, List, Set, Optional

# ── Metric Functions ──────────────────────────────────────────────────────────

def precision_at_k(retrieved, relevant, k):
    return sum(1 for d in retrieved[:k] if d in relevant) / k if k > 0 else 0.0

def recall_at_k(retrieved, relevant, k):
    return sum(1 for d in retrieved[:k] if d in relevant) / len(relevant) if relevant else 0.0

def dcg_at_k(retrieved, rel_map, k):
    s = 0.0
    for i, d in enumerate(retrieved[:k]):
        r = rel_map.get(d, 0)
        if r > 0:
            s += ((2**r) - 1) / math.log2(i + 2)
    return s

def idcg_at_k(rel_map, k):
    s = 0.0
    for i, r in enumerate(sorted(rel_map.values(), reverse=True)[:k]):
        if r > 0:
            s += ((2**r) - 1) / math.log2(i + 2)
    return s

def ndcg_at_k(retrieved, rel_map, k):
    ideal = idcg_at_k(rel_map, k)
    return dcg_at_k(retrieved, rel_map, k) / ideal if ideal > 0 else None

def ap_at_k(retrieved, relevant, k):
    if not relevant: return 0.0
    s, hits = 0.0, 0
    for i, d in enumerate(retrieved[:k]):
        if d in relevant:
            hits += 1
            s += hits / (i + 1)
    return s / len(relevant)

def mrr_at_k(retrieved, relevant, k):
    for i, d in enumerate(retrieved[:k]):
        if d in relevant:
            return 1.0 / (i + 1)
    return 0.0


# ── Full Pipeline for ONE query ──────────────────────────────────────────────

def run_single_query(
    query_text: str,
    query_uid: str,
    indexer,
    cross_encoder,
    colbert_reranker,
    conn: sqlite3.Connection,
    uid_lookup: dict,
    k: int = 10,
) -> List[str]:
    """
    Runs the COMPLETE Clinsight pipeline for a single query.
    Returns list of top-K patient_uids.
    """
    import numpy as np

    # ── Step 1: Clinical Parser (NER + Negation Removal) ──
    from app.services.query_improvement import query_improver
    improvement = query_improver._rule_normalize(query_text)
    dense_query = improvement

    # Sanitize for FTS5
    sanitized = query_improver._sanitize_for_fts5(dense_query)
    sparse_query = " ".join(sanitized.split()[:50])

    # ── Step 2: FAISS Dense Retrieval (top 100) ──
    dense_ids = indexer.search(dense_query, top_k=100)

    # ── Step 3: FTS5 Lexical Retrieval (top 100) ──
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT rowid FROM cases_fts WHERE text MATCH ? ORDER BY rank LIMIT 100",
            (sparse_query,)
        )
        sparse_ids = [row[0] for row in cursor.fetchall()]
    except Exception as e:
        sparse_ids = []

    # ── Step 4: RRF Fusion ──
    rrf_scores = {}
    for rank, did in enumerate(dense_ids):
        rrf_scores[did] = rrf_scores.get(did, 0.0) + 1.0 / (60 + rank)
    for rank, did in enumerate(sparse_ids):
        rrf_scores[did] = rrf_scores.get(did, 0.0) + 1.0 / (60 + rank)

    sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    top_rrf_ids = [did for did, _ in sorted_rrf[:50]]

    if not top_rrf_ids:
        return []

    # Hydrate from SQLite
    placeholders = ",".join("?" for _ in top_rrf_ids)
    rows = conn.execute(
        f"SELECT id, patient_uid, text FROM cases WHERE id IN ({placeholders})",
        tuple(int(i) for i in top_rrf_ids),
    ).fetchall()
    id_to_info = {row[0]: (row[1], row[2]) for row in rows}

    candidate_list = []
    for fid in top_rrf_ids:
        info = id_to_info.get(int(fid))
        if info and info[0] != query_uid:
            candidate_list.append(info)  # (uid, text)

    if not candidate_list:
        return []

    candidate_uids = [uid for uid, _ in candidate_list]
    candidate_texts = [text for _, text in candidate_list]

    # ── Step 5: ColBERT Reranking (top 20) ──
    if colbert_reranker is not None:
        colbert_results = colbert_reranker.rerank(query_text, candidate_texts, top_k=20)
        text_to_uid = {t: u for u, t in candidate_list}
        cb_uids = [text_to_uid[t] for t, _ in colbert_results if t in text_to_uid]
        cb_texts = [t for t, _ in colbert_results if t in text_to_uid]
    else:
        cb_uids = candidate_uids[:20]
        cb_texts = candidate_texts[:20]

    # ── Step 6: Cross-Encoder Reranking (top K) ──
    if cross_encoder is not None and cb_texts:
        pairs = [[query_text, doc] for doc in cb_texts]
        scores = cross_encoder.predict(pairs)
        scored = sorted(zip(cb_uids, scores), key=lambda x: x[1], reverse=True)
        return [uid for uid, _ in scored[:k]]

    return cb_uids[:k]


# ── FAISS-only baseline ──────────────────────────────────────────────────────

def faiss_only(query_text, query_uid, indexer, uid_lookup, k=10):
    raw_ids = indexer.search(query_text, top_k=k + 5)
    uids = []
    for fid in raw_ids:
        u = uid_lookup.get(fid)
        if u and u != query_uid:
            uids.append(u)
        if len(uids) >= k:
            break
    return uids


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    K = 10
    NUM = 5
    SEED = 42
    random.seed(SEED)

    DB_PATH = "app/data/pmc_cases.db"

    print("=" * 70)
    print("  CLINSIGHT LOCAL EVALUATION — 5 Queries, Full Pipeline")
    print("=" * 70)

    # ── Load models ONE BY ONE ──
    print("\n[1/4] Loading FAISS indexer...")
    from app.scripts.incremental_indexer import IncrementalFaissIndexer
    indexer = IncrementalFaissIndexer()

    print("[2/4] Loading ColBERT reranker...")
    try:
        from app.services.colbert_reranker import ColBERTReranker
        colbert = ColBERTReranker()
    except Exception as e:
        print(f"  ColBERT failed: {e}")
        colbert = None

    print("[3/4] Loading Cross-Encoder...")
    try:
        from sentence_transformers import CrossEncoder as CE
        ce_path = "app/data/finetuned-cross-encoder"
        if os.path.exists(ce_path):
            cross_enc = CE(ce_path)
            print(f"  Loaded fine-tuned from {ce_path}")
        else:
            cross_enc = CE("cross-encoder/ms-marco-MiniLM-L-6-v2")
    except Exception as e:
        print(f"  Cross-Encoder failed: {e}")
        cross_enc = None

    # ── Build lookups ──
    print("[4/4] Building lookups...")
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT id, patient_uid FROM cases").fetchall()
    uid_lookup = {r[0]: r[1] for r in rows}

    # ── Select 5 queries with ground truth ──
    gt_rows = conn.execute(
        "SELECT id, patient_uid, text, similar_patients FROM cases "
        "WHERE similar_patients IS NOT NULL AND similar_patients != '{}'"
    ).fetchall()

    candidates = []
    for r in gt_rows:
        sp = json.loads(r[3])
        if len(sp) > 0:
            candidates.append({
                "uid": r[1], "text": r[2], "rel": sp
            })

    print(f"  Ground truth pool: {len(candidates):,}")
    selected = random.sample(candidates, min(NUM, len(candidates)))
    print(f"  Selected: {len(selected)} queries\n")

    # ── Evaluate ONE query at a time ──
    faiss_metrics = []
    full_metrics = []

    for i, q in enumerate(selected):
        print(f"{'─'*70}")
        print(f"  Query {i+1}/{NUM}: {q['text'][:80]}...")
        print(f"  Ground truth: {len(q['rel'])} relevant patients")
        t0 = time.time()

        # Stage A: FAISS only
        faiss_uids = faiss_only(q["text"], q["uid"], indexer, uid_lookup, K)
        rel_set = {u for u, s in q["rel"].items() if s >= 1}

        f_ndcg = ndcg_at_k(faiss_uids, q["rel"], K)
        f_recall = recall_at_k(faiss_uids, rel_set, K)
        f_map = ap_at_k(faiss_uids, rel_set, K)
        faiss_metrics.append({"ndcg": f_ndcg, "recall": f_recall, "map": f_map})

        # Stage B: Full pipeline
        full_uids = run_single_query(
            q["text"], q["uid"],
            indexer, cross_enc, colbert,
            conn, uid_lookup, K
        )

        p_ndcg = ndcg_at_k(full_uids, q["rel"], K)
        p_recall = recall_at_k(full_uids, rel_set, K)
        p_map = ap_at_k(full_uids, rel_set, K)
        full_metrics.append({"ndcg": p_ndcg, "recall": p_recall, "map": p_map})

        elapsed = time.time() - t0
        print(f"  FAISS:    NDCG={f_ndcg or 0:.4f}  Recall={f_recall:.4f}  MAP={f_map:.4f}")
        print(f"  Pipeline: NDCG={p_ndcg or 0:.4f}  Recall={p_recall:.4f}  MAP={p_map:.4f}")
        print(f"  Time: {elapsed:.1f}s")

    conn.close()

    # ── Aggregate ──
    def avg_metric(results, key):
        vals = [r[key] for r in results if r[key] is not None]
        return sum(vals) / len(vals) if vals else 0.0

    f_ndcg_avg = avg_metric(faiss_metrics, "ndcg")
    f_recall_avg = avg_metric(faiss_metrics, "recall")
    f_map_avg = avg_metric(faiss_metrics, "map")

    p_ndcg_avg = avg_metric(full_metrics, "ndcg")
    p_recall_avg = avg_metric(full_metrics, "recall")
    p_map_avg = avg_metric(full_metrics, "map")

    print(f"\n{'='*70}")
    print(f"  FINAL RESULTS: {NUM} queries | K={K}")
    print(f"{'='*70}")
    print(f"{'Stage':<30} {'NDCG@10':>9} {'Recall@10':>10} {'MAP@10':>9}")
    print(f"{'─'*60}")
    print(f"{'FAISS Dense Only':<30} {f_ndcg_avg:>9.4f} {f_recall_avg:>10.4f} {f_map_avg:>9.4f}")
    print(f"{'Full Clinsight Pipeline':<30} {p_ndcg_avg:>9.4f} {p_recall_avg:>10.4f} {p_map_avg:>9.4f}")

    if f_ndcg_avg > 0:
        delta = ((p_ndcg_avg - f_ndcg_avg) / f_ndcg_avg) * 100
        sign = "+" if delta >= 0 else ""
        print(f"\n  📈 Pipeline Improvement: {sign}{delta:.1f}% NDCG@10")
    print(f"{'='*70}")

    # Save results
    out = {
        "faiss": {"ndcg@10": f_ndcg_avg, "recall@10": f_recall_avg, "map@10": f_map_avg},
        "final": {"ndcg@10": p_ndcg_avg, "recall@10": p_recall_avg, "map@10": p_map_avg},
        "counts": {"queries_used": NUM, "seed": SEED, "k": K},
    }
    with open("app/data/ppr_eval_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n💾 Saved to app/data/ppr_eval_results.json")
