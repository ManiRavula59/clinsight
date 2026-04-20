import os
import json
from datasets import load_dataset
from app.scripts.incremental_indexer import IncrementalFaissIndexer

def run_ingestion():
    # Fetch via streaming to avoid massive Parquet conversion crashes
    dataset = load_dataset("zhengyun21/PMC-Patients", split="train", streaming=True)
    
    indexer = IncrementalFaissIndexer()
    
    chunk_size = 5000
    current_chunk = []
    total_processed = 0
    
    # We will process 500 cases for the local working MVP
    MAX_TO_PROCESS = 500 
    
    print(f"Beginning incremental FAISS indexing in chunks of {chunk_size}...")
    
    all_cases = []
    
    for row in dataset:
        if total_processed >= MAX_TO_PROCESS:
            break
            
        # The dataset contains 'patient' (text) and 'patient_uid'
        # We clean and normalize the text slightly before embedding
        text = row.get("patient", "").strip()
        uid = row.get("patient_uid", f"pmc_{total_processed}")
        if text:
            # We prefix with a prompt to help the dense encoder (MedCPT/all-MiniLM)
            formatted_text = f"Clinical Case: {text}"
            current_chunk.append(formatted_text)
            
            all_cases.append({
                "id": uid,
                "text": formatted_text
            })
            
            total_processed += 1
            
        if len(current_chunk) >= chunk_size:
            print(f"Adding batch of {chunk_size} to FAISS. Total processed: {total_processed}")
            indexer.embed_and_add_chunk(current_chunk)
            current_chunk = [] # Free memory immediately
            
    # Process any remainder
    if current_chunk:
        print(f"Adding final batch of {len(current_chunk)} to FAISS. Total processed: {total_processed}")
        indexer.embed_and_add_chunk(current_chunk)
        
    print(f"Saving {len(all_cases)} text cases to local dict: app/data/pmc_cases.json")
    with open("app/data/pmc_cases.json", "w") as f:
        json.dump(all_cases, f)
        
    print(f"Ingestion complete. Total cases in local FAISS index: {indexer.index.ntotal}")

if __name__ == "__main__":
    run_ingestion()
