"""
Clinsight — PubMedBert Fine-Tuning on PMC-Patients
Compatible with sentence-transformers v5.x (Colab default as of 2026)
---------------------------------------------------------------------
Cell 1:  (nothing to install — sentence-transformers 5.x is preinstalled)
Cell 2:  paste this entire script

After training (~1.5h on T4):
  - Download the 'finetuned-pubmedbert' folder from the Files panel
  - Place it in backend/app/data/finetuned-pubmedbert/
  - Update MODEL_NAME in incremental_indexer.py to point to this path
"""

import sqlite3, json, logging, math, random, warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# ── sentence-transformers v5 correct imports ──────────────────────────────────
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from datasets import Dataset

# ── Config ────────────────────────────────────────────────────────────────────
DB_PATH      = "pmc_cases.db"
MODEL_NAME   = "pritamdeka/S-PubMedBert-MS-MARCO"
OUTPUT_DIR   = "./finetuned-pubmedbert"
BATCH_SIZE   = 64       # T4 handles 64 comfortably
NUM_EPOCHS   = 3
EVAL_SAMPLES = 300
MAX_SEQ_LEN  = 256      # Cuts encoding time in half vs default 512
CHAR_LIMIT   = 1500     # ~256 tokens worth of text
EVAL_STEPS   = 1000

# ── Load data from SQLite ─────────────────────────────────────────────────────
logger.info(f"Loading data from {DB_PATH}...")
conn = sqlite3.connect(DB_PATH)
rows = conn.execute(
    "SELECT patient_uid, text, similar_patients FROM cases WHERE similar_patients IS NOT NULL"
).fetchall()
text_lookup = {
    uid: text
    for uid, text in conn.execute("SELECT patient_uid, text FROM cases").fetchall()
}
conn.close()
logger.info(f"Total cases: {len(text_lookup):,}  |  Cases with ground truth: {len(rows):,}")

random.shuffle(rows)
eval_rows  = rows[:EVAL_SAMPLES]
train_rows = rows[EVAL_SAMPLES:]

# ── Build training pairs (anchor, positive) ───────────────────────────────────
anchors, positives = [], []
for uid, text, sim_json in train_rows:
    try:
        for sim_uid, score in json.loads(sim_json).items():
            if score >= 1 and sim_uid in text_lookup:
                anchors.append(text[:CHAR_LIMIT])
                positives.append(text_lookup[sim_uid][:CHAR_LIMIT])
    except Exception:
        pass

logger.info(f"Training pairs: {len(anchors):,}")
# v5 requires a HuggingFace Dataset with columns named "anchor" and "positive"
train_dataset = Dataset.from_dict({"anchor": anchors, "positive": positives})

# ── Build evaluation set ──────────────────────────────────────────────────────
eval_queries, eval_corpus, eval_relevant_docs = {}, {}, {}
for uid, text, sim_json in eval_rows:
    try:
        eval_queries[uid] = text[:CHAR_LIMIT]
        relevant = set()
        for sim_uid, score in json.loads(sim_json).items():
            if score >= 1 and sim_uid in text_lookup:
                relevant.add(sim_uid)
                eval_corpus[sim_uid] = text_lookup[sim_uid][:CHAR_LIMIT]
        if relevant:
            eval_relevant_docs[uid] = relevant
    except Exception:
        pass

# Add random negatives to make evaluation realistic
all_uids = list(text_lookup.keys())
for uid in random.sample(all_uids, min(2000, len(all_uids))):
    if uid not in eval_corpus:
        eval_corpus[uid] = text_lookup[uid][:CHAR_LIMIT]

logger.info(f"Eval queries: {len(eval_queries):,}  |  Corpus size: {len(eval_corpus):,}")

# ── Load model ────────────────────────────────────────────────────────────────
logger.info(f"Loading model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)
model.max_seq_length = MAX_SEQ_LEN   # Critical speed optimization

# ── Loss: InfoNCE (matches Eq.7 Ldense in base paper) ────────────────────────
loss = MultipleNegativesRankingLoss(model)

# ── Evaluator ─────────────────────────────────────────────────────────────────
evaluator = InformationRetrievalEvaluator(
    queries=eval_queries,
    corpus=eval_corpus,
    relevant_docs=eval_relevant_docs,
    name="pmc-dev",
    mrr_at_k=[10],
    ndcg_at_k=[10],
    show_progress_bar=True,
)

# ── Training arguments (v5 SentenceTransformerTrainingArguments) ─────────────
total_steps  = (len(train_dataset) // BATCH_SIZE) * NUM_EPOCHS
warmup_steps = int(total_steps * 0.1)

training_args = SentenceTransformerTrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    warmup_steps=warmup_steps,
    fp16=True,                        # Mixed precision on T4
    bf16=False,
    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_strategy="steps",
    save_steps=EVAL_STEPS,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="pmc-dev_ndcg@10",
    greater_is_better=True,
    logging_steps=100,
    dataloader_num_workers=4,         # Parallel CPU data loading
    dataloader_pin_memory=True,       # Faster CPU→GPU transfer
)

logger.info("=" * 60)
logger.info("Clinsight PubMedBert Fine-Tuning")
logger.info(f"  Pairs     : {len(train_dataset):,}")
logger.info(f"  Epochs    : {NUM_EPOCHS}")
logger.info(f"  Batch     : {BATCH_SIZE}")
logger.info(f"  Steps     : {total_steps:,}")
logger.info(f"  Max SeqLen: {MAX_SEQ_LEN}")
logger.info("=" * 60)

# ── Train ─────────────────────────────────────────────────────────────────────
trainer = SentenceTransformerTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    loss=loss,
    evaluator=evaluator,
)

trainer.train()

# ── Save ──────────────────────────────────────────────────────────────────────
model.save_pretrained(OUTPUT_DIR)
logger.info(f"\n✅ Training complete! Model saved to: {OUTPUT_DIR}")
logger.info("Download the 'finetuned-pubmedbert' folder from the Files panel (left sidebar).")
