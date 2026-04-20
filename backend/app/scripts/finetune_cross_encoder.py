import os
import json
import torch
import random
from torch.utils.data import DataLoader
from sentence_transformers import CrossEncoder, InputExample
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
import math
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Config
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2" # Base model to fine-tune
TRAIN_BATCH_SIZE = 16
NUM_EPOCHS = 2
OUTPUT_PATH = "./clinsight-pmc-cross-encoder"
DATA_FILE = "../data/pmc_cases.json" # Adjust path as needed on AWS

def prepare_training_data():
    """
    Parses PMC-Patients data and creates positive/negative pairs for the CrossEncoder.
    In PMC-Patients, similar_patients are positives. We sample random cases as negatives.
    """
    logger.info("Loading PMC-Patients dataset...")
    try:
        with open(DATA_FILE, 'r') as f:
            cases = json.load(f)
    except Exception as e:
        logger.error(f"Could not load data: {e}")
        return [], []

    train_examples = []
    eval_examples = []
    
    # Simple split
    split_idx = int(len(cases) * 0.9)
    train_cases = cases[:split_idx]
    eval_cases = cases[split_idx:]

    def build_examples(dataset):
        examples = []
        for case in dataset:
            query = f"{case.get('patient', 'Patient')} presented with {case.get('disease', 'condition')}. {case.get('patient_uid', '')}"
            
            # Positives
            for sim_id in case.get('similar_patients', []):
                # Find the similar case text (simplified for script, usually requires full DB lookup)
                sim_case = next((c for c in cases if c['patient_uid'] == sim_id), None)
                if sim_case:
                    doc = sim_case.get('patient', '')
                    examples.append(InputExample(texts=[query, doc], label=1.0))
            
            # Negatives (Random sample)
            if len(cases) > 5:
                neg_cases = random.sample(cases, 3)
                for neg_case in neg_cases:
                    if neg_case['patient_uid'] not in case.get('similar_patients', []) and neg_case['patient_uid'] != case['patient_uid']:
                        doc = neg_case.get('patient', '')
                        examples.append(InputExample(texts=[query, doc], label=0.0))
        return examples

    logger.info("Building training examples...")
    train_examples = build_examples(train_cases)
    logger.info(f"Built {len(train_examples)} training pairs.")
    
    logger.info("Building evaluation examples...")
    eval_examples = build_examples(eval_cases)
    logger.info(f"Built {len(eval_examples)} evaluation pairs.")

    return train_examples, eval_examples

def main():
    logger.info(f"Initializing CrossEncoder with {MODEL_NAME}")
    # Initialize cross-encoder. num_labels=1 means regression/binary classification
    model = CrossEncoder(MODEL_NAME, num_labels=1, max_length=512)

    train_examples, eval_examples = prepare_training_data()
    
    if not train_examples:
        logger.error("No training examples generated. Exiting.")
        return

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=TRAIN_BATCH_SIZE)

    # Evaluator to track performance during training
    evaluator = CEBinaryClassificationEvaluator.from_input_examples(eval_examples, name='pmc-eval')

    warmup_steps = math.ceil(len(train_dataloader) * NUM_EPOCHS * 0.1) # 10% of train data for warm-up
    logger.info(f"Warmup steps: {warmup_steps}")

    logger.info("Starting Fine-Tuning...")
    # Train the model
    model.fit(
        train_dataloader=train_dataloader,
        evaluator=evaluator,
        epochs=NUM_EPOCHS,
        evaluation_steps=500,
        warmup_steps=warmup_steps,
        output_path=OUTPUT_PATH,
        use_amp=True # Use Automatic Mixed Precision for faster training on modern GPUs
    )
    
    logger.info(f"Training complete. Model saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
