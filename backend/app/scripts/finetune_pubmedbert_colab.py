"""
Clinsight — PubMedBert Fine-Tuning (memory-efficient, T4, 12GB RAM)
------------------------------------------------------------------
Run in ONE Colab cell after uploading pmc_cases.db.
"""
import sqlite3, json, logging, math, random, gc, warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from datasets import Dataset

# ── Hyperparameters (tuned for 12GB RAM / T4 16GB VRAM) ──────────────────────
DB_PATH       = "pmc_cases.db"
MODEL_NAME    = "pritamdeka/S-PubMedBert-MS-MARCO"
OUTPUT_DIR    = "./finetuned-pubmedbert"

BATCH_SIZE    = 32          # Safe for 12GB RAM
GRAD_ACCUM    = 2           # Effective batch = 64
NUM_EPOCHS    = 3
LEARNING_RATE = 2e-5
WEIGHT_DECAY  = 0.01
MAX_SEQ_LEN   = 128         # Halves tensor memory vs 256
CHAR_LIMIT    = 800         # ~128 tokens worth of text
MAX_PAIRS     = 50000       # Cap pairs to avoid OOM
EVAL_STEPS    = 500
TEST_SAMPLES  = 150
VAL_SAMPLES   = 200

# ── PASS 1: Scan ground truth, collect only needed UIDs ──────────────────────
logger.info("Pass 1: Scanning ground truth for needed UIDs...")
conn = sqlite3.connect(DB_PATH)

rows = conn.execute(
    "SELECT patient_uid, text, similar_patients FROM cases WHERE similar_patients IS NOT NULL"
).fetchall()
logger.info(f"Rows with ground truth: {len(rows):,}")

random.shuffle(rows)
test_rows  = rows[:TEST_SAMPLES]
val_rows   = rows[TEST_SAMPLES:TEST_SAMPLES + VAL_SAMPLES]
train_rows = rows[TEST_SAMPLES + VAL_SAMPLES:]

# Only collect UIDs that appear as positives — NOT all 250k
needed_uids = set()
for uid, text, sim_json in rows:
    needed_uids.add(uid)
    try:
        for sim_uid in json.loads(sim_json):
            needed_uids.add(sim_uid)
    except Exception:
        pass
logger.info(f"Needed UIDs: {len(needed_uids):,} (vs 250k total — saves ~6GB RAM)")

# ── PASS 2: Load ONLY needed texts ──────────────────────────────────────────
logger.info("Pass 2: Loading only needed texts...")
text_lookup = {}
cursor = conn.cursor()
# SQLite IN clause has a 999-variable limit, so batch the query
uid_list = list(needed_uids)
for i in range(0, len(uid_list), 900):
    batch = uid_list[i:i+900]
    ph = ",".join("?" * len(batch))
    cursor.execute(f"SELECT patient_uid, text FROM cases WHERE patient_uid IN ({ph})", batch)
    for uid, text in cursor.fetchall():
        text_lookup[uid] = text[:CHAR_LIMIT]

conn.close()
del needed_uids, uid_list
gc.collect()
logger.info(f"Loaded {len(text_lookup):,} texts (memory-efficient)")

# ── Build Training Pairs (capped) ────────────────────────────────────────────
anchors, positives = [], []
for uid, text, sim_json in train_rows:
    if len(anchors) >= MAX_PAIRS:
        break
    try:
        for sim_uid, score in json.loads(sim_json).items():
            if score >= 1 and sim_uid in text_lookup and len(anchors) < MAX_PAIRS:
                anchors.append(text[:CHAR_LIMIT])
                positives.append(text_lookup[sim_uid])
    except Exception:
        pass

logger.info(f"Training pairs: {len(anchors):,} (cap: {MAX_PAIRS:,})")
train_dataset = Dataset.from_dict({"anchor": anchors, "positive": positives})
del anchors, positives, train_rows
gc.collect()

# ── Build Evaluators ──────────────────────────────────────────────────────────
def build_evaluator(split_rows, name, neg_count=500):
    queries, corpus, relevant_docs = {}, {}, {}
    for uid, text, sim_json in split_rows:
        try:
            queries[uid] = text[:CHAR_LIMIT]
            relevant = set()
            for sim_uid, score in json.loads(sim_json).items():
                if score >= 1 and sim_uid in text_lookup:
                    relevant.add(sim_uid)
                    corpus[sim_uid] = text_lookup[sim_uid]
            if relevant:
                relevant_docs[uid] = relevant
        except Exception:
            pass
    sample_uids = random.sample(list(text_lookup.keys()), min(neg_count, len(text_lookup)))
    for uid in sample_uids:
        if uid not in corpus:
            corpus[uid] = text_lookup[uid]
    logger.info(f"  {name}: {len(queries)} queries | {len(corpus)} corpus")
    return InformationRetrievalEvaluator(
        queries=queries, corpus=corpus, relevant_docs=relevant_docs,
        name=name, mrr_at_k=[10], ndcg_at_k=[10], show_progress_bar=True,
    )

val_evaluator  = build_evaluator(val_rows,  "val",  500)
test_evaluator = build_evaluator(test_rows, "test", 500)
del val_rows, test_rows, rows
gc.collect()

# ── Model + Loss ──────────────────────────────────────────────────────────────
logger.info(f"Loading model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)
model.max_seq_length = MAX_SEQ_LEN
loss = MultipleNegativesRankingLoss(model)

# ── Training Args ─────────────────────────────────────────────────────────────
total_steps  = (len(train_dataset) // (BATCH_SIZE * GRAD_ACCUM)) * NUM_EPOCHS
warmup_steps = int(total_steps * 0.1)

training_args = SentenceTransformerTrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    warmup_steps=warmup_steps,
    lr_scheduler_type="cosine",
    fp16=True,
    bf16=False,
    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_strategy="steps",
    save_steps=EVAL_STEPS,
    save_total_limit=1,             # Keep only 1 checkpoint to save disk
    load_best_model_at_end=True,
    metric_for_best_model="val_ndcg@10",
    greater_is_better=True,
    logging_steps=100,
    dataloader_num_workers=2,       # 2 workers, not 4 — saves RAM
    dataloader_pin_memory=False,    # Disabled — saves ~1GB RAM
)

logger.info("=" * 55)
logger.info(f"  Train pairs  : {len(train_dataset):,}")
logger.info(f"  Batch (eff.) : {BATCH_SIZE}x{GRAD_ACCUM} = {BATCH_SIZE*GRAD_ACCUM}")
logger.info(f"  Steps        : {total_steps:,}")
logger.info(f"  Max seq len  : {MAX_SEQ_LEN}")
logger.info("=" * 55)

# ── Train ─────────────────────────────────────────────────────────────────────
trainer = SentenceTransformerTrainer(
    model=model, args=training_args,
    train_dataset=train_dataset, loss=loss, evaluator=val_evaluator,
)
trainer.train()

# ── Final honest test score ───────────────────────────────────────────────────
logger.info("\nFinal test evaluation...")
test_results = test_evaluator(model)
logger.info(f"Test NDCG@10: {test_results:.4f}  (base paper BM25: ~0.27)")

model.save_pretrained(OUTPUT_DIR)
logger.info(f"\n✅ Done! Download '{OUTPUT_DIR}' from Files panel.")
