import sqlite3
import os
import json
from huggingface_hub import hf_hub_download
from tqdm import tqdm

DB_PATH = "app/data/pmc_cases.db"

def build_sqlite_db():
    print("Downloading PMC-Patients dataset from Hugging Face Hub...")
    print("This might take a moment to download (~500MB+)...")
    
    # Download the exact raw JSON to bypass PyArrow type-casting bugs
    file_path = hf_hub_download(repo_id="zhengyun21/PMC-Patients", filename="PMC-Patients-V2.json", repo_type="dataset")
    
    print("Loading JSON into memory...")
    with open(file_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
        
    total_cases = len(dataset)
    print(f"Successfully loaded {total_cases} cases.")
    
    # Ensure data directory exists
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    # Connect to SQLite (creates the file if it doesn't exist)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    print("Initializing SQLite Document Store...")
    # Create the cases table
    # We'll use the automatically generated index as the Primary Key integer ID
    # to perfectly align with FAISS which relies on 0-indexed integers.
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS cases (
            id INTEGER PRIMARY KEY,
            patient_uid TEXT UNIQUE,
            text TEXT
        )
    ''')
    
    # Clear any existing data if re-running
    cursor.execute('DELETE FROM cases')
    conn.commit()
    
    print(f"Ingesting {total_cases} cases into SQLite...")
    
    # Pre-compile the insert statement for speed
    insert_sql = 'INSERT INTO cases (id, patient_uid, text) VALUES (?, ?, ?)'
    
    # Batch processing is much faster for SQLite
    batch_size = 5000
    batch_data = []
    
    for i, row in enumerate(tqdm(dataset, desc="Writing to SQLite")):
        # Clean the text gently just in case
        text = row.get("patient", "").strip()
        uid = row.get("patient_uid", f"unknown_{i}")
        
        # We force the ID to be exactly 'i' to match FAISS
        batch_data.append((i, uid, text))
        
        if len(batch_data) >= batch_size:
            cursor.executemany(insert_sql, batch_data)
            conn.commit()
            batch_data = []
            
    # Insert any remaining records
    if batch_data:
        cursor.executemany(insert_sql, batch_data)
        conn.commit()
        
    print(f"Finished building Document Store at {DB_PATH}")
    conn.close()

if __name__ == "__main__":
    build_sqlite_db()
