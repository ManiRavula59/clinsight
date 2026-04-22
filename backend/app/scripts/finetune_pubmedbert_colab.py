"""
Clinsight — Full Dataset Chunked Training
==========================================
Trains on ALL pairs by loading 8000 at a time.
Upload pmc_cases.db to Drive, then run this.

How it works:
  - Reads 8000 training pairs from SQLite (using OFFSET)
  - Trains 1 epoch on that chunk
  - Moves to next 8000 pairs
  - Repeats until entire dataset is covered
  - Then does a second full pass (epoch 2)
  - RAM never exceeds ~4GB
"""

# ================== MOUNT DRIVE ==================
from google.colab import drive
drive.mount('/content/drive')

import sqlite3, json, logging, random, gc, os, torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
torch.backends.cuda.matmul.allow_tf32 = True

# ================== CONFIG ==================
DB_PATH    = "/content/drive/MyDrive/pmc_cases.db"
MODEL_PATH = "/content/drive/MyDrive/finetuned-pubmedbert"
OUTPUT_DIR = "/content/drive/MyDrive/finetuned-pubmedbert"

CHUNK_SIZE  = 8000    # Pairs per chunk (fits in 12GB RAM)
TEXT_LEN    = 300
FULL_PASSES = 3       # 3 full passes = 3 epochs over entire dataset
BATCH_SIZE  = 16
GRAD_ACCUM  = 2

print("GPU:", torch.cuda.is_available())
assert os.path.exists(DB_PATH), "DB not found!"

# ================== COUNT TOTAL GROUND TRUTH ROWS ==================
conn = sqlite3.connect(DB_PATH)
total_rows = conn.execute("SELECT COUNT(*) FROM cases WHERE similar_patients != '{}'").fetchone()[0]
logger.info(f"Total ground truth rows: {total_rows:,}")
conn.close()

# ================== LOAD MODEL (from previous training) ==================
if os.path.exists(MODEL_PATH) and os.path.exists(os.path.join(MODEL_PATH, "config.json")):
    logger.info(f"Loading fine-tuned model from: {MODEL_PATH}")
    model = SentenceTransformer(MODEL_PATH, device="cuda" if torch.cuda.is_available() else "cpu")
else:
    logger.info("No fine-tuned model found — loading base PubMedBert")
    model = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO", device="cuda" if torch.cuda.is_available() else "cpu")

model.max_seq_length = 128

# ================== CHUNKED TRAINING LOOP ==================
chunk_id = 0

for full_pass in range(FULL_PASSES):
    logger.info(f"\n{'='*60}")
    logger.info(f"  FULL PASS {full_pass + 1} / {FULL_PASSES}")
    logger.info(f"{'='*60}")

    offset = 0

    while offset < total_rows:
        chunk_id += 1
        logger.info(f"\n--- Chunk {chunk_id} | Rows {offset} to {offset + CHUNK_SIZE} ---")

        # Open fresh connection per chunk
        conn = sqlite3.connect(DB_PATH)

        # Stream rows for this chunk only
        pair_buf = []
        cursor = conn.execute(
            "SELECT id, patient_uid, text, similar_patients FROM cases "
            "WHERE similar_patients != '{}' LIMIT ? OFFSET ?",
            (CHUNK_SIZE * 3, offset)  # Read extra rows since not all have valid pairs
        )

        for rid, uid, txt, sj in cursor:
            try:
                for su, sc in json.loads(sj).items():
                    if sc >= 1:
                        pair_buf.append((txt[:TEXT_LEN], su))
                        if len(pair_buf) >= CHUNK_SIZE:
                            break
            except:
                pass
            if len(pair_buf) >= CHUNK_SIZE:
                break

        if not pair_buf:
            logger.info("No more pairs — moving to next pass")
            conn.close()
            break

        # Fetch positive texts
        uids = list(set(u for _, u in pair_buf))
        pos_text = {}
        for i in range(0, len(uids), 300):
            b = uids[i:i+300]
            for u, t in conn.execute(
                f"SELECT patient_uid, text FROM cases WHERE patient_uid IN ({','.join('?'*len(b))})", b
            ).fetchall():
                pos_text[u] = t[:TEXT_LEN]

        conn.close()
        del uids

        # Build final pairs
        anchors, positives = [], []
        for a, u in pair_buf:
            if u in pos_text:
                anchors.append(a)
                positives.append(pos_text[u])

        del pair_buf, pos_text
        gc.collect()

        if len(anchors) < 100:
            logger.info(f"Too few pairs ({len(anchors)}) — skipping chunk")
            offset += CHUNK_SIZE * 3
            del anchors, positives
            gc.collect()
            continue

        logger.info(f"Training on {len(anchors):,} pairs...")
        ds = Dataset.from_dict({"anchor": anchors, "positive": positives}).shuffle(seed=chunk_id)
        del anchors, positives
        gc.collect()

        # Train this chunk (1 epoch per chunk)
        loss = MultipleNegativesRankingLoss(model)
        steps_this_chunk = len(ds) // (BATCH_SIZE * GRAD_ACCUM)

        args = SentenceTransformerTrainingArguments(
            output_dir=OUTPUT_DIR,
            num_train_epochs=1,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            learning_rate=2e-5,
            weight_decay=0.01,
            warmup_steps=max(1, int(steps_this_chunk * 0.1)),
            lr_scheduler_type="cosine",
            fp16=True if torch.cuda.is_available() else False,
            logging_steps=50,
            save_strategy="no",
            dataloader_num_workers=2,
            dataloader_pin_memory=True,
        )

        trainer = SentenceTransformerTrainer(
            model=model, args=args, train_dataset=ds, loss=loss,
        )
        trainer.train()

        # Save after each chunk (to Drive — survives disconnects)
        model.save_pretrained(OUTPUT_DIR)
        logger.info(f"✅ Chunk {chunk_id} done — saved to Drive")

        # Cleanup
        del ds, loss, trainer, args
        gc.collect()
        torch.cuda.empty_cache()

        # Move offset
        offset += CHUNK_SIZE * 3

print(f"\n🎉 Full training complete! {chunk_id} chunks trained across {FULL_PASSES} passes.")
print(f"Model saved at: {OUTPUT_DIR}")
