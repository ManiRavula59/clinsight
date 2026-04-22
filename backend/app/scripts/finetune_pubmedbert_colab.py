"""
Clinsight — PubMedBert Fine-Tuning (sentence-transformers v5, T4 optimized)
---------------------------------------------------------------------------
Run in ONE Colab cell after uploading pmc_cases.db.
"""
import sqlite3, json, logging, math, random, warnings
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

# ── Hyperparameters ───────────────────────────────────────────────────────────
DB_PATH         = "pmc_cases.db"
MODEL_NAME      = "pritamdeka/S-PubMedBert-MS-MARCO"
OUTPUT_DIR      = "./finetuned-pubmedbert"

BATCH_SIZE      = 64        # T4 16GB handles this well
GRAD_ACCUM      = 2         # Effective batch = 64 * 2 = 128 (better generalization)
NUM_EPOCHS      = 3         # 3 epochs: enough for fine-tune, avoids overfitting
LEARNING_RATE   = 2e-5      # Standard BERT fine-tune LR
WEIGHT_DECAY    = 0.01      # L2 regularization — prevents overfitting
MAX_SEQ_LEN     = 256       # 2x faster than default 512, minimal accuracy loss
CHAR_LIMIT      = 1500      # ~256 tokens of text

# Data splits (no strict 70/20/10 needed for contrastive learning)
TEST_SAMPLES    = 200       # Held-out test set (honest final score)
VAL_SAMPLES     = 300       # Validation during training (early stopping)
# Remaining rows → Training

EVAL_STEPS      = 500       # Evaluate every 500 steps

# ── Load Data ─────────────────────────────────────────────────────────────────
logger.info("Loading data from SQLite...")
conn = sqlite3.connect(DB_PATH)
rows = conn.execute(
    "SELECT patient_uid, text, similar_patients FROM cases WHERE similar_patients IS NOT NULL"
).fetchall()
text_lookup = {uid: t for uid, t in conn.execute("SELECT patient_uid, text FROM cases").fetchall()}
conn.close()
logger.info(f"Total: {len(text_lookup):,} cases | {len(rows):,} with ground truth labels")

# ── Three-way split: test / val / train ───────────────────────────────────────
random.shuffle(rows)
test_rows  = rows[:TEST_SAMPLES]
val_rows   = rows[TEST_SAMPLES:TEST_SAMPLES + VAL_SAMPLES]
train_rows = rows[TEST_SAMPLES + VAL_SAMPLES:]
logger.info(f"Split → Train: {len(train_rows):,} | Val: {len(val_rows):,} | Test: {len(test_rows):,}")

# ── Build Training Pairs ───────────────────────────────────────────────────────
# InfoNCE / MultipleNegativesRankingLoss: only positive pairs needed.
# In-batch negatives are generated automatically from other examples in the batch.
anchors, positives = [], []
for uid, text, sim_json in train_rows:
    try:
        for sim_uid, score in json.loads(sim_json).items():
            if score >= 1 and sim_uid in text_lookup:
                anchors.append(text[:CHAR_LIMIT])
                positives.append(text_lookup[sim_uid][:CHAR_LIMIT])
    except Exception:
        pass
logger.info(f"Training pairs built: {len(anchors):,}")
train_dataset = Dataset.from_dict({"anchor": anchors, "positive": positives})

# ── Build Validation Evaluator ─────────────────────────────────────────────────
def build_ir_evaluator(split_rows, name):
    queries, corpus, relevant_docs = {}, {}, {}
    for uid, text, sim_json in split_rows:
        try:
            queries[uid] = text[:CHAR_LIMIT]
            relevant = set()
            for sim_uid, score in json.loads(sim_json).items():
                if score >= 1 and sim_uid in text_lookup:
                    relevant.add(sim_uid)
                    corpus[sim_uid] = text_lookup[sim_uid][:CHAR_LIMIT]
            if relevant:
                relevant_docs[uid] = relevant
        except Exception:
            pass
    # Add random negatives for a realistic retrieval difficulty
    for uid in random.sample(list(text_lookup.keys()), min(2000, len(text_lookup))):
        if uid not in corpus:
            corpus[uid] = text_lookup[uid][:CHAR_LIMIT]
    logger.info(f"{name}: {len(queries):,} queries | {len(corpus):,} corpus size")
    return InformationRetrievalEvaluator(
        queries=queries, corpus=corpus, relevant_docs=relevant_docs,
        name=name, mrr_at_k=[10], ndcg_at_k=[10],
        show_progress_bar=True,
    )

val_evaluator  = build_ir_evaluator(val_rows,  name="val")
test_evaluator = build_ir_evaluator(test_rows, name="test")

# ── Load Model ─────────────────────────────────────────────────────────────────
logger.info(f"Loading: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)
model.max_seq_length = MAX_SEQ_LEN   # Critical for T4 speed

# ── Loss ───────────────────────────────────────────────────────────────────────
loss = MultipleNegativesRankingLoss(model)

# ── Training Arguments (all speed + regularization knobs set) ─────────────────
total_steps  = (len(train_dataset) // (BATCH_SIZE * GRAD_ACCUM)) * NUM_EPOCHS
warmup_steps = int(total_steps * 0.1)

training_args = SentenceTransformerTrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,       # Effective batch = 128
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,                    # L2 regularization
    warmup_steps=warmup_steps,
    lr_scheduler_type="cosine",                   # Cosine decay = smooth convergence
    fp16=True,                                    # Mixed precision on T4
    bf16=False,
    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_strategy="steps",
    save_steps=EVAL_STEPS,
    save_total_limit=2,
    load_best_model_at_end=True,                  # Auto early stopping
    metric_for_best_model="val_ndcg@10",
    greater_is_better=True,
    logging_steps=50,
    dataloader_num_workers=4,                     # Parallel data loading
    dataloader_pin_memory=True,                   # Faster CPU->GPU
    torch_compile=False,                          # Set True if Colab has torch>=2.0
)

# ── Summary ───────────────────────────────────────────────────────────────────
logger.info("=" * 60)
logger.info(f"  Training pairs    : {len(train_dataset):,}")
logger.info(f"  Effective batch   : {BATCH_SIZE * GRAD_ACCUM} (64 x {GRAD_ACCUM} accum)")
logger.info(f"  Total steps       : {total_steps:,}")
logger.info(f"  Warmup steps      : {warmup_steps:,}")
logger.info(f"  LR                : {LEARNING_RATE} (cosine decay)")
logger.info(f"  Weight decay (L2) : {WEIGHT_DECAY}")
logger.info(f"  Max seq length    : {MAX_SEQ_LEN} tokens")
logger.info(f"  FP16 mixed prec.  : True")
logger.info("=" * 60)

# ── Train ─────────────────────────────────────────────────────────────────────
trainer = SentenceTransformerTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    loss=loss,
    evaluator=val_evaluator,         # Monitor val NDCG@10 during training
)
trainer.train()

# ── Final Test Evaluation (honest held-out score) ─────────────────────────────
logger.info("\nRunning FINAL evaluation on held-out test set...")
test_results = test_evaluator(model)
logger.info(f"Test NDCG@10 : {test_results:.4f}")
logger.info(f"(Compare with base paper BM25 baseline: ~0.27 NDCG@10)")

# ── Save ──────────────────────────────────────────────────────────────────────
model.save_pretrained(OUTPUT_DIR)
logger.info(f"\n✅ Done! Model saved to: {OUTPUT_DIR}")
logger.info("Download 'finetuned-pubmedbert' from Files panel (left sidebar).")
logger.info("Place it in backend/app/data/ and update MODEL_NAME in incremental_indexer.py")
