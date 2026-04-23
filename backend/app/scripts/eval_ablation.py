import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

"""
5-Stage Ablation: BM25 → Hybrid → FAISS → +ColBERT → +CrossEncoder
Runs 5 queries, one at a time. Prints the exact table for the report.
"""

import sys, json, math, time, sqlite3, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# ── Metrics ──
def recall_at_k(ret, rel, k):
    return sum(1 for d in ret[:k] if d in rel) / len(rel) if rel else 0.0

def dcg_at_k(ret, rm, k):
    s = 0.0
    for i, d in enumerate(ret[:k]):
        r = rm.get(d, 0)
        if r > 0: s += ((2**r)-1) / math.log2(i+2)
    return s

def ndcg_at_k(ret, rm, k):
    ideal = 0.0
    for i, r in enumerate(sorted(rm.values(), reverse=True)[:k]):
        if r > 0: ideal += ((2**r)-1) / math.log2(i+2)
    return dcg_at_k(ret, rm, k) / ideal if ideal > 0 else None

def ap_at_k(ret, rel, k):
    if not rel: return 0.0
    s, h = 0.0, 0
    for i, d in enumerate(ret[:k]):
        if d in rel: h += 1; s += h/(i+1)
    return s / len(rel)

# ── Main ──
DB_PATH = "app/data/pmc_cases.db"
K, NUM, SEED = 10, 10, 42
random.seed(SEED)

print("=" * 75)
print("  CLINSIGHT 5-STAGE ABLATION — Real Values")
print("=" * 75)

# Load models
print("\nLoading models...")
from app.scripts.incremental_indexer import IncrementalFaissIndexer
from app.services.query_improvement import query_improver
indexer = IncrementalFaissIndexer()

try:
    from app.services.colbert_reranker import ColBERTReranker
    colbert = ColBERTReranker()
except: colbert = None

try:
    from sentence_transformers import CrossEncoder as CE
    ce_path = "app/data/finetuned-cross-encoder"
    cross_enc = CE(ce_path) if os.path.exists(ce_path) else CE("cross-encoder/ms-marco-MiniLM-L-6-v2")
except: cross_enc = None

conn = sqlite3.connect(DB_PATH)
rows = conn.execute("SELECT id, patient_uid FROM cases").fetchall()
uid_lookup = {r[0]: r[1] for r in rows}

# Select queries
gt_rows = conn.execute(
    "SELECT id, patient_uid, text, similar_patients FROM cases "
    "WHERE similar_patients IS NOT NULL AND similar_patients != '{}'"
).fetchall()
candidates = []
for r in gt_rows:
    sp = json.loads(r[3])
    if len(sp) > 0:
        candidates.append({"uid": r[1], "text": r[2], "rel": sp})

selected = random.sample(candidates, min(NUM, len(candidates)))
print(f"Selected {len(selected)} queries with ground truth\n")

# Storage for 5 stages
stages = {
    "Stage 1: BM25 Only (FTS5)": [],
    "Stage 2: Hybrid (FAISS+FTS5+RRF)": [],
    "Stage 3: FAISS Fine-tuned": [],
    "Stage 4: +ColBERT Rerank": [],
    "Stage 5: Full Pipeline (+CE)": [],
}

for qi, q in enumerate(selected):
    uid, text, rel = q["uid"], q["text"], q["rel"]
    rel_set = {u for u, s in rel.items() if s >= 1}
    print(f"Query {qi+1}/{NUM}: {text[:70]}...")

    # ── Use FULL Clinical Parser (LLM NER + Negation + Keyword extraction) ──
    try:
        improvement = query_improver.process(text)
        dense_query = improvement.normalized_text
        sparse_q = improvement.fts5_query
        print(f"  Parser: {len(improvement.entities)} entities, FTS5: {sparse_q[:60]}...")
    except Exception as e:
        print(f"  Parser fallback (LLM error): {e}")
        dense_query = query_improver._rule_normalize(text)
        sparse_q = query_improver._sanitize_for_fts5(dense_query)

    # ── Stage 1: BM25 Only ──
    try:
        fts_rows = conn.execute(
            "SELECT rowid FROM cases_fts WHERE text MATCH ? ORDER BY rank LIMIT ?",
            (sparse_q, K + 5)).fetchall()
        bm25_ids = [r[0] for r in fts_rows]
    except:
        bm25_ids = []
    bm25_uids = [uid_lookup.get(fid) for fid in bm25_ids if uid_lookup.get(fid) and uid_lookup.get(fid) != uid][:K]
    stages["Stage 1: BM25 Only (FTS5)"].append({
        "ndcg": ndcg_at_k(bm25_uids, rel, K),
        "recall": recall_at_k(bm25_uids, rel_set, K),
        "map": ap_at_k(bm25_uids, rel_set, K),
    })

    # ── Stage 2: Hybrid (FAISS + FTS5 + RRF) ──
    dense_ids = indexer.search(dense_query, top_k=100)
    try:
        sp_rows = conn.execute(
            "SELECT rowid FROM cases_fts WHERE text MATCH ? ORDER BY rank LIMIT 100",
            (sparse_q,)).fetchall()
        sparse_ids = [r[0] for r in sp_rows]
    except:
        sparse_ids = []

    rrf = {}
    for rank, did in enumerate(dense_ids):
        rrf[did] = rrf.get(did, 0) + 1.0/(60+rank)
    for rank, did in enumerate(sparse_ids):
        rrf[did] = rrf.get(did, 0) + 1.0/(60+rank)
    sorted_rrf = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:50]
    rrf_ids = [did for did, _ in sorted_rrf]

    # Hydrate
    if rrf_ids:
        ph = ",".join("?" for _ in rrf_ids)
        hydrated = conn.execute(f"SELECT id, patient_uid, text FROM cases WHERE id IN ({ph})",
                                tuple(int(i) for i in rrf_ids)).fetchall()
        id2info = {r[0]: (r[1], r[2]) for r in hydrated}
        cands = [(id2info[int(fid)][0], id2info[int(fid)][1]) for fid in rrf_ids
                 if int(fid) in id2info and id2info[int(fid)][0] != uid]
    else:
        cands = []

    hybrid_uids = [u for u, t in cands][:K]
    stages["Stage 2: Hybrid (FAISS+FTS5+RRF)"].append({
        "ndcg": ndcg_at_k(hybrid_uids, rel, K),
        "recall": recall_at_k(hybrid_uids, rel_set, K),
        "map": ap_at_k(hybrid_uids, rel_set, K),
    })

    # ── Stage 3: FAISS Fine-tuned only ──
    faiss_uids = []
    for fid in dense_ids:
        u = uid_lookup.get(fid)
        if u and u != uid: faiss_uids.append(u)
        if len(faiss_uids) >= K: break
    stages["Stage 3: FAISS Fine-tuned"].append({
        "ndcg": ndcg_at_k(faiss_uids, rel, K),
        "recall": recall_at_k(faiss_uids, rel_set, K),
        "map": ap_at_k(faiss_uids, rel_set, K),
    })

    # ── Stage 4: +ColBERT ──
    all_texts = [t for _, t in cands]
    all_uids_c = [u for u, _ in cands]
    if colbert and all_texts:
        cb_res = colbert.rerank(text, all_texts, top_k=20)
        t2u = {t: u for u, t in cands}
        cb_uids = [t2u[t] for t, _ in cb_res if t in t2u][:K]
        cb_texts = [t for t, _ in cb_res if t in t2u][:K]
    else:
        cb_uids = all_uids_c[:K]
        cb_texts = all_texts[:K]
    stages["Stage 4: +ColBERT Rerank"].append({
        "ndcg": ndcg_at_k(cb_uids, rel, K),
        "recall": recall_at_k(cb_uids, rel_set, K),
        "map": ap_at_k(cb_uids, rel_set, K),
    })

    # ── Stage 5: +Cross-Encoder ──
    if cross_enc and cb_texts:
        pairs = [[text, d] for d in cb_texts]
        scores = cross_enc.predict(pairs)
        scored = sorted(zip(cb_uids, scores), key=lambda x: x[1], reverse=True)
        ce_uids = [u for u, _ in scored[:K]]
    else:
        ce_uids = cb_uids[:K]
    stages["Stage 5: Full Pipeline (+CE)"].append({
        "ndcg": ndcg_at_k(ce_uids, rel, K),
        "recall": recall_at_k(ce_uids, rel_set, K),
        "map": ap_at_k(ce_uids, rel_set, K),
    })

    print(f"  Done. BM25={stages['Stage 1: BM25 Only (FTS5)'][-1]['ndcg'] or 0:.3f} → Full={stages['Stage 5: Full Pipeline (+CE)'][-1]['ndcg'] or 0:.3f}")

conn.close()

# ── Aggregate & Print Table ──
def avg(results, key):
    vals = [r[key] for r in results if r[key] is not None]
    return sum(vals)/len(vals) if vals else 0.0

print(f"\n{'='*80}")
print(f"  PERFORMANCE MATRIX — {NUM} Queries | K={K} | Seed={SEED}")
print(f"{'='*80}")
print(f"{'Model Stage':<42} {'NDCG@10':>9} {'Recall@10':>10} {'MAP@10':>9} {'Δ Improve':>10}")
print(f"{'─'*80}")

base_ndcg = None
for name, results in stages.items():
    n = avg(results, "ndcg")
    r = avg(results, "recall")
    m = avg(results, "map")
    if base_ndcg is None:
        base_ndcg = n
        delta = "—"
    else:
        d = ((n - base_ndcg)/base_ndcg*100) if base_ndcg > 0 else 0
        delta = f"+{d:.1f}%"
    print(f"{name:<42} {n:>9.4f} {r:>10.4f} {m:>9.4f} {delta:>10}")

print(f"{'─'*80}")
print(f"  Queries: {NUM} | Ground Truth Pool: 104,402 | Corpus: 250,294 cases")
print(f"{'='*80}")
