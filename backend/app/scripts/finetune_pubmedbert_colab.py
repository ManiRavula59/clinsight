"""
Google Colab Fine-Tuning Script for S-PubMedBert (Bi-Encoder)
Compatible with sentence-transformers v3+
-------------------------------------------------------------
Instructions for Google Colab:
1. Upload `pmc_cases.db` to Colab (Files panel on the left)
2. Run Cell 1:  !pip install -q sentence-transformers datasets torch
3. Run Cell 2:  paste this entire script and run
4. After training, download the `finetuned-pubmedbert` folder
5. Place it in your Clinsight backend directory
"""

import sqlite3
import json
import logging
import math
import random
import os

# ── New sentence-transformers v3 API imports (no deprecation warnings) ──
from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from datasets import Dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
DB_PATH       = "pmc_cases.db"                      # Must be uploaded to Colab
MODEL_NAME    = "pritamdeka/S-PubMedBert-MS-MARCO"  # 768-dim medical bi-encoder
OUTPUT_DIR    = "./finetuned-pubmedbert"
BATCH_SIZE    = 16          # Increase to 32 on A100
NUM_EPOCHS    = 3
EVAL_SAMPLES  = 500
WARMUP_RATIO  = 0.1


def load_data_from_db():
    """Load PMC-Patients ground truth pairs from SQLite database."""
    logger.info(f"Connecting to database: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)

    # Load cases with ground truth similarity labels
    rows = conn.execute(
        "SELECT patient_uid, text, similar_patients FROM cases WHERE similar_patients IS NOT NULL"
    ).fetchall()

    # Build a fast UID → text lookup
    logger.info("Building text lookup dictionary...")
    text_lookup = {}
    for uid, text in conn.execute("SELECT patient_uid, text FROM cases").fetchall():
        text_lookup[uid] = text

    logger.info(f"Total cases: {len(text_lookup):,}  |  Cases with ground truth: {len(rows):,}")

    # Split into train/eval
    random.shuffle(rows)
    eval_rows  = rows[:EVAL_SAMPLES]
    train_rows = rows[EVAL_SAMPLES:]

    # ── Build Training Pairs ──────────────────────────────────────────────────
    # MultipleNegativesRankingLoss expects (anchor, positive) pairs.
    # The batch itself provides hard in-batch negatives automatically.
    anchors, positives = [], []
    for uid, text, sim_json in train_rows:
        try:
            similarities = json.loads(sim_json)
        except Exception:
            continue
        for sim_uid, score in similarities.items():
            if score >= 1 and sim_uid in text_lookup:   # rel=1 or rel=2 → positive
                anchors.append(text)
                positives.append(text_lookup[sim_uid])

    logger.info(f"Training pairs built: {len(anchors):,}")

    # ── Build Evaluation Set ──────────────────────────────────────────────────
    eval_queries, eval_corpus, eval_relevant_docs = {}, {}, {}
    for uid, text, sim_json in eval_rows:
        try:
            similarities = json.loads(sim_json)
        except Exception:
            continue
        eval_queries[uid] = text
        relevant = set()
        for sim_uid, score in similarities.items():
            if score >= 1 and sim_uid in text_lookup:
                relevant.add(sim_uid)
                eval_corpus[sim_uid] = text_lookup[sim_uid]
        if relevant:
            eval_relevant_docs[uid] = relevant

    # Add random negatives to eval corpus for realistic evaluation difficulty
    all_uids = list(text_lookup.keys())
    for _ in range(2000):
        rand_uid = random.choice(all_uids)
        if rand_uid not in eval_corpus:
            eval_corpus[rand_uid] = text_lookup[rand_uid]

    conn.close()

    # Convert to HuggingFace Dataset (required by SentenceTransformerTrainer)
    train_dataset = Dataset.from_dict({"anchor": anchors, "positive": positives})
    logger.info(f"Train dataset: {len(train_dataset)} rows  |  Eval queries: {len(eval_queries)}")

    return train_dataset, eval_queries, eval_corpus, eval_relevant_docs


def main():
    # ── 1. Load Data ──────────────────────────────────────────────────────────
    train_dataset, eval_queries, eval_corpus, eval_relevant_docs = load_data_from_db()

    if len(train_dataset) == 0:
        logger.error("No training data found. Check that similar_patients column is populated.")
        return

    # ── 2. Load Base Model ────────────────────────────────────────────────────
    logger.info(f"Loading base model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    # ── 3. Loss Function ──────────────────────────────────────────────────────
    # InfoNCE / MultipleNegativesRankingLoss = matches eq.(7) Ldense(θ) in base paper
    loss = MultipleNegativesRankingLoss(model)

    # ── 4. Evaluator ─────────────────────────────────────────────────────────
    evaluator = InformationRetrievalEvaluator(
        queries=eval_queries,
        corpus=eval_corpus,
        relevant_docs=eval_relevant_docs,
        name="pmc-dev",
        mrr_at_k=[10],
        ndcg_at_k=[10],
        accuracy_at_k=[1, 10],
        show_progress_bar=True,
    )

    # ── 5. Training Arguments ─────────────────────────────────────────────────
    total_steps = (len(train_dataset) // BATCH_SIZE) * NUM_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)

    args = SentenceTransformerTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        warmup_steps=warmup_steps,
        fp16=True,                        # Mixed precision — essential on T4/A100
        bf16=False,
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        logging_steps=100,
        run_name="clinsight-pubmedbert-finetuning",
    )

    # ── 6. Train ──────────────────────────────────────────────────────────────
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=loss,
        evaluator=evaluator,
    )

    logger.info("=" * 60)
    logger.info("Starting Fine-Tuning — S-PubMedBert on PMC-Patients")
    logger.info(f"  Training pairs  : {len(train_dataset):,}")
    logger.info(f"  Epochs          : {NUM_EPOCHS}")
    logger.info(f"  Batch size      : {BATCH_SIZE}")
    logger.info(f"  Total steps     : {total_steps:,}")
    logger.info(f"  Warmup steps    : {warmup_steps:,}")
    logger.info("=" * 60)

    trainer.train()

    # ── 7. Save Final Model ───────────────────────────────────────────────────
    model.save_pretrained(OUTPUT_DIR)
    logger.info(f"\n✅ Fine-tuning complete! Model saved to: {OUTPUT_DIR}")
    logger.info("Download the 'finetuned-pubmedbert' folder and place it in your backend directory.")
    logger.info("Then update MODEL_NAME in incremental_indexer.py to point to this local path.")


if __name__ == "__main__":
    main()
