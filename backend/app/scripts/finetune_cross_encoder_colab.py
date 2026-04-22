"""
Clinsight — Cross-Encoder Fine-Tuning on PMC-Patients
=====================================================
Run AFTER bi-encoder training completes.
Uses same chunked approach — fits in 12GB RAM.

Cross-encoder takes (query, document) pairs and outputs a relevance score.
We fine-tune ms-marco-MiniLM-L-6-v2 on PMC-Patients ground truth.
"""

# ================== MOUNT DRIVE ==================
from google.colab import drive
drive.mount('/content/drive')

import sqlite3, json, logging, random, gc, os, torch
import numpy as np
from datasets import Dataset
from sentence_transformers import CrossEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
torch.backends.cuda.matmul.allow_tf32 = True

# ================== CONFIG ==================
DB_PATH     = "/content/drive/MyDrive/pmc_cases.db"
MODEL_NAME  = "cross-encoder/ms-marco-MiniLM-L-6-v2"
OUTPUT_DIR  = "/content/drive/MyDrive/finetuned-cross-encoder"

CHUNK_SIZE  = 15000    # MiniLM is tiny — can handle big chunks
TEXT_LEN    = 500      # Longer text = better context for cross-encoder
FULL_PASSES = 3
BATCH_SIZE  = 64       # MiniLM-L6 is only 22M params — 64 fits easily on T4
GRAD_ACCUM  = 2        # Effective batch = 128
NUM_EPOCHS_PER_CHUNK = 1

print("GPU:", torch.cuda.is_available())
assert os.path.exists(DB_PATH), "DB not found!"

# ================== COUNT ROWS ==================
conn = sqlite3.connect(DB_PATH)
total_rows = conn.execute("SELECT COUNT(*) FROM cases WHERE similar_patients != '{}'").fetchone()[0]
logger.info(f"Ground truth rows: {total_rows:,}")
conn.close()

# ================== LOAD MODEL ==================
if os.path.exists(OUTPUT_DIR) and os.path.exists(os.path.join(OUTPUT_DIR, "config.json")):
    logger.info(f"Loading fine-tuned cross-encoder from: {OUTPUT_DIR}")
    model = CrossEncoder(OUTPUT_DIR, device="cuda" if torch.cuda.is_available() else "cpu")
else:
    logger.info(f"Loading base model: {MODEL_NAME}")
    model = CrossEncoder(MODEL_NAME, device="cuda" if torch.cuda.is_available() else "cpu")

model.model.config.max_length = 256

# ================== CHUNKED TRAINING ==================
chunk_id = 0

for full_pass in range(FULL_PASSES):
    logger.info(f"\n{'='*60}")
    logger.info(f"  FULL PASS {full_pass + 1} / {FULL_PASSES}")
    logger.info(f"{'='*60}")

    offset = 0

    while offset < total_rows:
        chunk_id += 1
        logger.info(f"\n--- Chunk {chunk_id} | Offset {offset} ---")

        conn = sqlite3.connect(DB_PATH)

        # Collect positive pairs from this chunk
        pos_pairs = []
        cursor = conn.execute(
            "SELECT patient_uid, text, similar_patients FROM cases "
            "WHERE similar_patients != '{}' LIMIT ? OFFSET ?",
            (CHUNK_SIZE * 3, offset)
        )

        needed_uids = set()
        raw_pairs = []

        for uid, txt, sj in cursor:
            try:
                sims = json.loads(sj)
                for sim_uid, score in sims.items():
                    if score >= 1:
                        raw_pairs.append((txt[:TEXT_LEN], sim_uid, float(score)))
                        needed_uids.add(sim_uid)
                        if len(raw_pairs) >= CHUNK_SIZE:
                            break
            except:
                pass
            if len(raw_pairs) >= CHUNK_SIZE:
                break

        if not raw_pairs:
            logger.info("No more pairs — next pass")
            conn.close()
            break

        # Fetch positive texts
        pos_text = {}
        uid_list = list(needed_uids)
        for i in range(0, len(uid_list), 300):
            b = uid_list[i:i+300]
            for u, t in conn.execute(
                f"SELECT patient_uid, text FROM cases WHERE patient_uid IN ({','.join('?'*len(b))})", b
            ).fetchall():
                pos_text[u] = t[:TEXT_LEN]

        # Fetch random negatives (same number as positives)
        neg_texts = []
        for u, t in conn.execute(
            "SELECT patient_uid, text FROM cases ORDER BY RANDOM() LIMIT ?",
            (min(len(raw_pairs), CHUNK_SIZE),)
        ).fetchall():
            neg_texts.append(t[:TEXT_LEN])

        conn.close()
        del needed_uids, uid_list
        gc.collect()

        # Build training data: positives (label=1) + negatives (label=0)
        sentences_1 = []  # query texts
        sentences_2 = []  # document texts
        labels = []       # 1 = relevant, 0 = irrelevant

        # Positive pairs
        for query_txt, sim_uid, score in raw_pairs:
            if sim_uid in pos_text:
                sentences_1.append(query_txt)
                sentences_2.append(pos_text[sim_uid])
                # Normalize score: rel=2 → 1.0, rel=1 → 0.7
                labels.append(1.0 if score >= 2 else 0.7)

        # Negative pairs (random documents paired with queries)
        num_neg = min(len(sentences_1), len(neg_texts))
        for i in range(num_neg):
            sentences_1.append(sentences_1[i % len(sentences_1)])  # Reuse query
            sentences_2.append(neg_texts[i])
            labels.append(0.0)

        del raw_pairs, pos_text, neg_texts
        gc.collect()

        if len(sentences_1) < 100:
            logger.info(f"Too few pairs ({len(sentences_1)}) — skipping")
            offset += CHUNK_SIZE * 3
            del sentences_1, sentences_2, labels
            gc.collect()
            continue

        logger.info(f"Training: {len(sentences_1):,} pairs ({sum(1 for l in labels if l > 0)} pos, {sum(1 for l in labels if l == 0)} neg)")

        # Shuffle together
        combined = list(zip(sentences_1, sentences_2, labels))
        random.shuffle(combined)
        sentences_1 = [c[0] for c in combined]
        sentences_2 = [c[1] for c in combined]
        labels = [c[2] for c in combined]
        del combined

        # Train cross-encoder
        from sentence_transformers.cross_encoder.trainer import CrossEncoderTrainer
        from sentence_transformers.cross_encoder.training_args import CrossEncoderTrainingArguments

        train_dataset = Dataset.from_dict({
            "sentence1": sentences_1,
            "sentence2": sentences_2,
            "label": labels,
        })

        del sentences_1, sentences_2, labels
        gc.collect()

        args = CrossEncoderTrainingArguments(
            output_dir=OUTPUT_DIR,
            num_train_epochs=NUM_EPOCHS_PER_CHUNK,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            learning_rate=2e-5,
            weight_decay=0.01,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            fp16=True if torch.cuda.is_available() else False,
            logging_steps=50,
            save_strategy="no",
            dataloader_num_workers=4,     # 8GB RAM free = room for workers
            dataloader_pin_memory=True,   # Faster GPU transfer
        )

        trainer = CrossEncoderTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
        )
        trainer.train()

        # Save to Drive after each chunk
        model.save_pretrained(OUTPUT_DIR)
        logger.info(f"✅ Chunk {chunk_id} done — saved to Drive")

        del train_dataset, trainer, args
        gc.collect()
        torch.cuda.empty_cache()

        offset += CHUNK_SIZE * 3

print(f"\n🎉 Cross-encoder training complete! {chunk_id} chunks across {FULL_PASSES} passes.")
print(f"Model saved at: {OUTPUT_DIR}")
print("Download and place in backend/app/data/finetuned-cross-encoder/")
