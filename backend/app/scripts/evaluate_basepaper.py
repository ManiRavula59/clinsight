"""
evaluate_ppr_at10.py
=====================
Authentic Patient-Patient Retrieval (PPR) Evaluation Module for Clinsight.

Uses REAL ground truth from PMC-Patients `similar_patients` field.
Evaluates at K=10 (matching the base paper's NDCG@10 primary metric).

Stage-wise evaluation:
  Stage A — FAISS retrieval quality (initial dense retrieval)
  Stage B — Final reranked quality (after ColBERT + Cross-Encoder)

Ground truth:
  For each query patient q:
    similar_patients[q] = {patient_uid: score}
    where score=2 (highly similar) or score=1 (moderately similar)

Metrics (all @10):
  1. Recall@10       — fraction of relevant items found in top-10
  2. Precision@10    — fraction of top-10 that are relevant
  3. DCG@10          — graded relevance (score 2 > score 1)
  4. NDCG@10         — normalized DCG (primary metric, matches base paper)
  5. AP@10           — average precision (binary relevance)
  6. MAP@10          — mean AP over all queries
  7. MRR@10          — reciprocal rank of first relevant result

Usage:
  python app/scripts/evaluate_ppr_at10.py [--num-queries 500] [--output results.json]
"""

import math
import json
import sqlite3
import random
import time
import argparse
import os
import sys
from typing import Dict, List, Set, Tuple, Optional

# ──────────────────────────────────────────────────────────────────────────────
# METRIC FUNCTIONS (deterministic, no LLM)
# ──────────────────────────────────────────────────────────────────────────────

def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Precision@K = |retrieved[:k] ∩ relevant| / k
    What fraction of our top-k results are actually relevant?
    """
    if k <= 0:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for doc in top_k if doc in relevant)
    return hits / k


def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Recall@K = |retrieved[:k] ∩ relevant| / |relevant|
    What fraction of ALL relevant items did we find in top-k?
    """
    if len(relevant) == 0:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for doc in top_k if doc in relevant)
    return hits / len(relevant)


def dcg_at_k(retrieved: List[str], relevance_map: Dict[str, int], k: int) -> float:
    """
    DCG@K = Σ_{i=1..k} (2^{rel_i} - 1) / log2(i+1)

    Uses graded relevance: rel=2 (highly similar), rel=1 (moderate), rel=0 (irrelevant).
    Documents with score=2 contribute (2^2-1)/(log2) = 3/log2 per position.
    Documents with score=1 contribute (2^1-1)/(log2) = 1/log2 per position.
    """
    score = 0.0
    for i, doc in enumerate(retrieved[:k]):
        rel = relevance_map.get(doc, 0)
        if rel > 0:
            # gain = 2^rel - 1, discount = log2(i+2) because i is 0-indexed
            gain = (2 ** rel) - 1
            discount = math.log2(i + 2)  # i+2 because i=0 → position 1 → log2(2)
            score += gain / discount
    return score


def idcg_at_k(relevance_map: Dict[str, int], k: int) -> float:
    """
    IDCG@K = DCG of the ideal ranking.
    Sort all relevant docs by score descending, take top k, compute DCG.
    """
    # Get all relevance scores, sorted descending
    sorted_rels = sorted(relevance_map.values(), reverse=True)
    # Truncate to k
    ideal_rels = sorted_rels[:k]

    score = 0.0
    for i, rel in enumerate(ideal_rels):
        if rel > 0:
            gain = (2 ** rel) - 1
            discount = math.log2(i + 2)
            score += gain / discount
    return score


def ndcg_at_k(retrieved: List[str], relevance_map: Dict[str, int], k: int) -> Optional[float]:
    """
    NDCG@K = DCG@K / IDCG@K
    Returns None if IDCG@K == 0 (query excluded from NDCG averaging).
    """
    ideal = idcg_at_k(relevance_map, k)
    if ideal == 0.0:
        return None  # Undefined — exclude from averaging
    actual = dcg_at_k(retrieved, relevance_map, k)
    return actual / ideal


def ap_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    AP@K = (1/|G|) * Σ_{i=1..k} Precision@i * I(d_i ∈ G)

    Average Precision: rewards systems that place relevant items higher.
    """
    if len(relevant) == 0:
        return 0.0

    score = 0.0
    hits = 0
    for i, doc in enumerate(retrieved[:k]):
        if doc in relevant:
            hits += 1
            precision_i = hits / (i + 1)
            score += precision_i

    return score / len(relevant)


def mrr_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    MRR@K = 1 / rank_of_first_relevant_in_top_k
    Returns 0 if no relevant result in top-k.
    """
    for i, doc in enumerate(retrieved[:k]):
        if doc in relevant:
            return 1.0 / (i + 1)
    return 0.0


# ──────────────────────────────────────────────────────────────────────────────
# ALL METRICS IN ONE CALL
# ──────────────────────────────────────────────────────────────────────────────

def compute_all_metrics(
    retrieved_uids: List[str],
    relevance_map: Dict[str, int],
    k: int = 10,
) -> Dict[str, Optional[float]]:
    """
    Computes all 7 metrics for a single query at K.

    Args:
        retrieved_uids: Ranked list of patient_uids (top result first).
        relevance_map:  {patient_uid: score} where score ∈ {1, 2}.
        k:              Evaluation cutoff (default 10).

    Returns:
        Dict with keys: recall, precision, dcg, ndcg (or None), ap, mrr
    """
    # Binary relevance set: all uids with score >= 1
    relevant_set = {uid for uid, score in relevance_map.items() if score >= 1}

    return {
        "recall": recall_at_k(retrieved_uids, relevant_set, k),
        "precision": precision_at_k(retrieved_uids, relevant_set, k),
        "dcg": dcg_at_k(retrieved_uids, relevance_map, k),
        "ndcg": ndcg_at_k(retrieved_uids, relevance_map, k),  # None if IDCG=0
        "ap": ap_at_k(retrieved_uids, relevant_set, k),
        "mrr": mrr_at_k(retrieved_uids, relevant_set, k),
    }


# ──────────────────────────────────────────────────────────────────────────────
# RETRIEVAL FUNCTIONS (FAISS + Reranker)
# ──────────────────────────────────────────────────────────────────────────────

def get_faiss_top_n(
    indexer, query_text: str, query_uid: str, n: int, uid_lookup: Dict[int, str]
) -> List[str]:
    """
    Runs FAISS dense retrieval and returns top-N patient_uids.
    Removes the query itself from results.
    """
    raw_ids = indexer.search(query_text, top_k=n + 5)  # Over-fetch to compensate for self-removal
    result_uids = []
    for fid in raw_ids:
        uid = uid_lookup.get(fid)
        if uid and uid != query_uid:
            result_uids.append(uid)
        if len(result_uids) >= n:
            break
    return result_uids


def get_final_top_k(
    indexer,
    cross_encoder,
    colbert_reranker,
    query_text: str,
    query_uid: str,
    k: int,
    uid_lookup: Dict[int, str],
    id_lookup: Dict[str, int],
    conn: sqlite3.Connection,
) -> List[str]:
    """
    Runs full pipeline: FAISS → FTS5 → RRF → ColBERT → Cross-Encoder.
    Returns top-K patient_uids after all reranking stages.
    Removes the query itself from results.
    """
    import numpy as np

    from app.services.query_improvement import query_improver
    
    # Pre-process query to expand shortcuts (yo -> year old) and sanitize for FTS5
    # Force bypass of LLM API to prevent timeouts during offline 30-batch run
    improvement = query_improver._rule_normalize(query_text)
    dense_query = improvement
    
    # Truncate sparse query to top 50 words to avoid FTS5 hanging
    sanitized = query_improver._sanitize_for_fts5(dense_query)
    sparse_query = " ".join(sanitized.split()[:50])

    # Stage 1: FAISS Dense (top 100)
    dense_ids = indexer.search(dense_query, top_k=100)

    # Stage 2: FTS5 BM25 (top 100)
    sparse_ids = indexer.search_bm25(sparse_query, top_k=100)

    # Stage 3: RRF Fusion
    rrf_scores = {}
    for rank, doc_idx in enumerate(dense_ids):
        rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + (1.0 / (60 + rank))
    for rank, doc_idx in enumerate(sparse_ids):
        rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + (1.0 / (60 + rank))

    sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    top_rrf_ids = [doc_idx for doc_idx, _ in sorted_rrf[:50]]

    # Hydrate texts from SQLite
    if not top_rrf_ids:
        return []
    placeholders = ",".join("?" for _ in top_rrf_ids)
    rows = conn.execute(
        f"SELECT id, patient_uid, text FROM cases WHERE id IN ({placeholders})",
        tuple(int(i) for i in top_rrf_ids),
    ).fetchall()
    id_to_info = {row[0]: (row[1], row[2]) for row in rows}

    # Maintain RRF order
    candidate_list = []
    for fid in top_rrf_ids:
        info = id_to_info.get(int(fid))
        if info:
            uid, text = info
            if uid != query_uid:  # Remove self
                candidate_list.append((uid, text))

    if not candidate_list:
        return []

    candidate_uids = [uid for uid, _ in candidate_list]
    candidate_texts = [text for _, text in candidate_list]

    # Stage 4: ColBERT Rerank (→ top 20)
    if colbert_reranker is not None:
        colbert_results = colbert_reranker.rerank(query_text, candidate_texts, top_k=20)
        text_to_uid = {text: uid for uid, text in candidate_list}
        colbert_uids = [text_to_uid[t] for t, _ in colbert_results if t in text_to_uid]
        colbert_texts = [t for t, _ in colbert_results if t in text_to_uid]
    else:
        colbert_uids = candidate_uids[:20]
        colbert_texts = candidate_texts[:20]

    # Stage 5: Cross-Encoder Rerank (→ top k)
    if cross_encoder is not None and colbert_texts:
        cross_inp = [[query_text, doc] for doc in colbert_texts]
        scores = cross_encoder.predict(cross_inp)
        scored = list(zip(colbert_uids, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [uid for uid, _ in scored[:k]]

    return colbert_uids[:k]


# ──────────────────────────────────────────────────────────────────────────────
# AGGREGATION ENGINE
# ──────────────────────────────────────────────────────────────────────────────

def aggregate_metrics(
    per_query_results: List[Dict[str, Optional[float]]],
) -> Dict[str, float]:
    """
    Averages per-query metrics. 
    For NDCG: only queries where NDCG is not None (IDCG > 0) are included.
    """
    n = len(per_query_results)
    if n == 0:
        return {k: 0.0 for k in ["recall@10", "precision@10", "dcg@10", "ndcg@10", "map@10", "mrr@10"]}

    sums = {"recall": 0.0, "precision": 0.0, "dcg": 0.0, "ap": 0.0, "mrr": 0.0}
    ndcg_sum = 0.0
    ndcg_count = 0

    for qr in per_query_results:
        sums["recall"] += qr["recall"]
        sums["precision"] += qr["precision"]
        sums["dcg"] += qr["dcg"]
        sums["ap"] += qr["ap"]
        sums["mrr"] += qr["mrr"]
        if qr["ndcg"] is not None:
            ndcg_sum += qr["ndcg"]
            ndcg_count += 1

    return {
        "recall@10": sums["recall"] / n,
        "precision@10": sums["precision"] / n,
        "dcg@10": sums["dcg"] / n,
        "ndcg@10": ndcg_sum / ndcg_count if ndcg_count > 0 else 0.0,
        "map@10": sums["ap"] / n,
        "mrr@10": sums["mrr"] / n,
        "ndcg_queries_used": ndcg_count,
    }


# ──────────────────────────────────────────────────────────────────────────────
# LIVE QUERY EVALUATOR (used by agent.py at runtime)
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_single_query(
    retrieved_uids: List[str],
    query_uid: str,
    conn: sqlite3.Connection,
    k: int = 10,
) -> Dict[str, Optional[float]]:
    """
    Evaluates a single query's retrieved results against real ground truth.
    Called by agent.py during live queries.

    Args:
        retrieved_uids: Ranked list of patient_uids returned by our pipeline.
        query_uid:      The query patient's UID.
        conn:           SQLite connection to pmc_cases.db.
        k:              Evaluation cutoff.

    Returns:
        Dict with all metric values, or all zeros if no ground truth.
    """
    # Fetch ground truth for this query
    row = conn.execute(
        "SELECT similar_patients FROM cases WHERE patient_uid = ?", (query_uid,)
    ).fetchone()

    if not row or not row[0] or row[0] == "{}":
        # No ground truth for this query — return zeros
        return {
            "recall": 0.0, "precision": 0.0, "dcg": 0.0,
            "ndcg": None, "ap": 0.0, "mrr": 0.0,
            "has_ground_truth": False,
        }

    relevance_map = json.loads(row[0])

    # Remove query from retrieved list (safety check)
    clean_retrieved = [uid for uid in retrieved_uids if uid != query_uid][:k]

    metrics = compute_all_metrics(clean_retrieved, relevance_map, k)
    metrics["has_ground_truth"] = True
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# BATCH EVALUATION (offline benchmarking)
# ──────────────────────────────────────────────────────────────────────────────

def run_batch_evaluation(num_queries: int = 500, seed: int = 42, output_path: str = None):
    """
    Runs full offline batch evaluation over a random sample of queries.
    Compares FAISS-only vs Full-pipeline (FAISS+FTS5+ColBERT+CrossEncoder).
    """
    random.seed(seed)
    K = 10
    DB_PATH = "app/data/pmc_cases.db"

    print("=" * 70)
    print("  CLINSIGHT PPR EVALUATION — Authentic Ground Truth @ K=10")
    print("=" * 70)

    # ── Load models ──────────────────────────────────────────────────────
    print("\n[1/5] Loading models...")
    from app.scripts.incremental_indexer import IncrementalFaissIndexer
    from sentence_transformers import CrossEncoder as CE

    indexer = IncrementalFaissIndexer()

    colbert = None
    cross_enc = None

    # ── Build lookup tables ──────────────────────────────────────────────
    print("[2/5] Building UID ↔ FAISS-ID lookups...")
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT id, patient_uid FROM cases").fetchall()
    uid_lookup = {row[0]: row[1] for row in rows}  # faiss_id → uid
    id_lookup = {row[1]: row[0] for row in rows}   # uid → faiss_id

    # ── Select evaluation queries ────────────────────────────────────────
    print("[3/5] Selecting evaluation queries with ground truth...")
    gt_rows = conn.execute(
        "SELECT id, patient_uid, text, similar_patients FROM cases "
        "WHERE similar_patients IS NOT NULL AND similar_patients != '{}'"
    ).fetchall()

    # Filter to queries that have at least 1 relevant patient
    eval_candidates = []
    for row in gt_rows:
        sp = json.loads(row[3])
        if len(sp) > 0:
            eval_candidates.append({
                "faiss_id": row[0],
                "uid": row[1],
                "text": row[2],
                "relevance_map": sp,
            })

    print(f"  Total candidates with ground truth: {len(eval_candidates):,}")
    selected = random.sample(eval_candidates, min(num_queries, len(eval_candidates)))
    print(f"  Selected for evaluation: {len(selected)}")

    # ── Run evaluation ───────────────────────────────────────────────────
    print(f"[4/5] Evaluating {len(selected)} queries (FAISS vs Hybrid BasePaper)...")
    faiss_results = []
    final_results = []
    per_query_log = []

    start_time = time.time()
    for i, q in enumerate(selected):
        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  Query {i+1}/{len(selected)}  ({rate:.1f} q/s)")

        q_uid = q["uid"]
        q_text = q["text"]
        rel_map = q["relevance_map"]

        # Stage A: FAISS only
        faiss_uids = get_faiss_top_n(indexer, q_text, q_uid, K, uid_lookup)
        faiss_m = compute_all_metrics(faiss_uids, rel_map, K)
        faiss_results.append(faiss_m)

        # Stage B: Base Paper Hybrid (FAISS + FTS5 RRF)
        final_uids = get_final_top_k(
            indexer, cross_enc, colbert,
            q_text, q_uid, K,
            uid_lookup, id_lookup, conn,
        )
        final_m = compute_all_metrics(final_uids, rel_map, K)
        final_results.append(final_m)

        # Per-query log
        per_query_log.append({
            "q_uid": q_uid,
            "faiss": {k: v for k, v in faiss_m.items()},
            "final": {k: v for k, v in final_m.items()},
        })

    total_time = time.time() - start_time
    print(f"  Done in {total_time:.1f}s ({len(selected)/total_time:.1f} queries/sec)")

    conn.close()

    # ── Aggregate ────────────────────────────────────────────────────────
    print("[5/5] Aggregating results...")
    faiss_agg = aggregate_metrics(faiss_results)
    final_agg = aggregate_metrics(final_results)

    delta = {}
    for key in ["recall@10", "precision@10", "dcg@10", "ndcg@10", "map@10", "mrr@10"]:
        delta[key] = final_agg[key] - faiss_agg[key]

    # ── Report ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"  RESULTS: {len(selected)} queries | K=10 | Seed={seed}")
    print("=" * 70)
    print(f"{'Metric':<16} {'FAISS':>10} {'Hybrid(BP)':>10} {'Δ (gain)':>10}")
    print("-" * 50)
    for key in ["recall@10", "precision@10", "ndcg@10", "map@10", "mrr@10", "dcg@10"]:
        f_val = faiss_agg[key]
        fn_val = final_agg[key]
        d_val = delta[key]
        sign = "+" if d_val >= 0 else ""
        print(f"{key:<16} {f_val:>10.4f} {fn_val:>10.4f} {sign}{d_val:>9.4f}")
    print("-" * 50)
    print(f"NDCG queries:    {faiss_agg.get('ndcg_queries_used', 'N/A')}")
    print(f"Total queries:   {len(selected)}")
    print()

    # ── Save outputs ─────────────────────────────────────────────────────
    summary = {
        "faiss": {k: v for k, v in faiss_agg.items() if k != "ndcg_queries_used"},
        "final": {k: v for k, v in final_agg.items() if k != "ndcg_queries_used"},
        "delta": delta,
        "counts": {
            "queries_total": len(eval_candidates),
            "queries_used": len(selected),
            "queries_used_ndcg": faiss_agg.get("ndcg_queries_used", 0),
            "seed": seed,
            "k": K,
            "eval_time_seconds": round(total_time, 1),
        },
    }

    out_path = output_path or "app/data/ppr_eval_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {out_path}")

    # Per-query JSONL
    jsonl_path = out_path.replace(".json", "_per_query.jsonl")
    with open(jsonl_path, "w") as f:
        for entry in per_query_log:
            f.write(json.dumps(entry) + "\n")
    print(f"Per-query log saved to: {jsonl_path}")

    return summary


# ──────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Ensure 'app' package is importable when running standalone
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

    parser = argparse.ArgumentParser(description="Clinsight PPR Evaluation @ K=10")
    parser.add_argument("--num-queries", type=int, default=500, help="Number of queries to evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    run_batch_evaluation(
        num_queries=args.num_queries,
        seed=args.seed,
        output_path=args.output,
    )
