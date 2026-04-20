"""
Google Colab Fine-Tuning Script for S-PubMedBert (Bi-Encoder)
-------------------------------------------------------------
Instructions for Google Colab:
1. Upload your `pmc_cases.db` file to the Colab environment.
2. Install required libraries: `!pip install sentence-transformers torch`
3. Run this script: `!python finetune_pubmedbert_colab.py`
4. Once completed, download the `finetuned-pubmedbert` folder and place it in your Clinsight backend.
"""

import sqlite3
import json
import logging
import math
import random
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Configuration ---
DB_PATH = "pmc_cases.db"  # Make sure this is uploaded to Colab
MODEL_NAME = "pritamdeka/S-PubMedBert-MS-MARCO"
OUTPUT_DIR = "./finetuned-pubmedbert"
BATCH_SIZE = 16
NUM_EPOCHS = 3
EVAL_SAMPLES = 500

def load_data_from_db():
    logger.info(f"Connecting to {DB_PATH}...")
    conn = sqlite3.connect(DB_PATH)
    
    # We only need rows that have similar_patients ground truth
    rows = conn.execute(
        "SELECT patient_uid, text, similar_patients FROM cases WHERE similar_patients IS NOT NULL"
    ).fetchall()
    
    # Create a quick lookup dictionary for texts
    logger.info("Building text lookup dictionary...")
    text_lookup = {}
    
    # Also fetch all IDs and texts to resolve positives
    all_rows = conn.execute("SELECT patient_uid, text FROM cases").fetchall()
    for uid, text in all_rows:
        text_lookup[uid] = text

    logger.info(f"Loaded {len(text_lookup)} total cases. Found {len(rows)} cases with ground truth.")

    train_examples = []
    eval_queries = {}
    eval_corpus = {}
    eval_relevant_docs = {}

    random.shuffle(rows)
    eval_rows = rows[:EVAL_SAMPLES]
    train_rows = rows[EVAL_SAMPLES:]

    # Build Training Examples
    logger.info("Building Training Pairs (Positive Pairs for Contrastive Loss)...")
    for uid, text, sim_json in train_rows:
        try:
            similarities = json.loads(sim_json)
        except Exception:
            continue
            
        for sim_uid, score in similarities.items():
            if score >= 1 and sim_uid in text_lookup:
                pos_text = text_lookup[sim_uid]
                # We use MultipleNegativesRankingLoss, so we only need positive pairs.
                # In-batch negatives are automatically used.
                train_examples.append(InputExample(texts=[text, pos_text]))

    # Build Eval Sets
    logger.info("Building Evaluation Set...")
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
            
    # Add some random negatives to the eval corpus to make it realistic
    all_uids = list(text_lookup.keys())
    for _ in range(2000):
        rand_uid = random.choice(all_uids)
        if rand_uid not in eval_corpus:
            eval_corpus[rand_uid] = text_lookup[rand_uid]

    conn.close()
    
    logger.info(f"Total Training Pairs: {len(train_examples)}")
    return train_examples, eval_queries, eval_corpus, eval_relevant_docs

def main():
    logger.info("Loading Data...")
    train_examples, eval_queries, eval_corpus, eval_relevant_docs = load_data_from_db()

    if not train_examples:
        logger.error("No training data found. Make sure the database has 'similar_patients' data.")
        return

    logger.info(f"Loading Base Model: {MODEL_NAME}")
    # Load the base model
    model = SentenceTransformer(MODEL_NAME)

    # Dataloader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)
    
    # Loss Function: Contrastive Softmax Loss (InfoNCE)
    # This exactly matches equation (7) Ldense(theta) in the base paper.
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    # Evaluator
    evaluator = InformationRetrievalEvaluator(
        queries=eval_queries,
        corpus=eval_corpus,
        relevant_docs=eval_relevant_docs,
        name='pmc-dev',
        mrr_at_k=[10],
        ndcg_at_k=[10],
        accuracy_at_k=[1, 10],
        show_progress_bar=True
    )

    # Warmup
    warmup_steps = math.ceil(len(train_dataloader) * NUM_EPOCHS * 0.1)
    
    logger.info("Starting Fine-Tuning Loop...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=NUM_EPOCHS,
        evaluation_steps=500,
        warmup_steps=warmup_steps,
        output_path=OUTPUT_DIR,
        use_amp=True, # Automatic Mixed Precision (very fast on Colab T4/A100)
        show_progress_bar=True
    )
    
    logger.info(f"Fine-tuning complete! Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
