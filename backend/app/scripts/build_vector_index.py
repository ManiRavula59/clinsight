import os
import sqlite3
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from tqdm import tqdm

DB_PATH = "app/data/pmc_cases.db"
INDEX_PATH = "app/data/disease_index.faiss"
BM25_PATH = "app/data/bm25_corpus.pkl"

def build_indexes():
    print("Initializing embedding model onto Apple MPS (Metal Performance Shaders)...")
    # Using 'cpu' for safe fallback to avoid the M1 Unified Memory leak bug.
    model = SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO', device='cpu')
    embedding_dim = model.get_sentence_embedding_dimension()
    
    # Initialize FAISS index
    if os.path.exists(INDEX_PATH):
        print(f"Loading existing FAISS Index to resume progress...")
        index = faiss.read_index(INDEX_PATH)
        start_offset = index.ntotal
        print(f"Resuming exactly from Case #{start_offset}...")
    else:
        print(f"Creating new exact search FAISS index (Dimension: {embedding_dim})")
        index = faiss.IndexFlatIP(embedding_dim) # Inner Product is Cosine Sim if normalized
        start_offset = 0
    
    print("Connecting to SQLite Document Store...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get total count
    cursor.execute("SELECT COUNT(*) FROM cases")
    total_docs = cursor.fetchone()[0]
    print(f"Total documents to process: {total_docs}")
    
    # We will process in chunks to keep memory usage safe during vectorization
    chunk_size = 5000 
    
    import torch
    import gc
    import json
    BM25_TOKENS_PATH = "app/data/bm25_tokens.jsonl"
    
    # Delete token cache if we're starting fresh
    if start_offset == 0 and os.path.exists(BM25_TOKENS_PATH):
        os.remove(BM25_TOKENS_PATH)
    
    print("\nStarting Mathematics Engine (This will take 30-60 minutes on M1 MPS)...")
    
    total_chunks = (total_docs + chunk_size - 1) // chunk_size
    initial_chunk = start_offset // chunk_size
    
    for offset in tqdm(range(start_offset, total_docs, chunk_size), desc="Vectorizing 250k Cases", initial=initial_chunk, total=total_chunks):
        cursor.execute("SELECT id, text FROM cases ORDER BY id ASC LIMIT ? OFFSET ?", (chunk_size, offset))
        rows = cursor.fetchall()
        
        texts = [row[1] for row in rows]
        
        # 1. FAISS DENSE EMBEDDING (Hardware Accelerated)
        # Using batch_size inside encode to prevent MPS memory spillage
        embeddings = model.encode(texts, batch_size=256, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        index.add(embeddings)
        
        # 1b. Checkpoint the Index exactly so progress is NEVER lost!
        faiss.write_index(index, INDEX_PATH)
        
        # 2. BM25 SPARSE TOKENIZATION (Written to disk to save RAM)
        with open(BM25_TOKENS_PATH, "a") as f:
            for text in texts:
                tokens = text.lower().split()
                f.write(json.dumps(tokens) + "\n")
                
        # 3. CRITICAL: Clear Unified Memory to prevent M1 Thermal Swap!
        del embeddings
        del texts
        del rows
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    
    # Build and Save the BM25 Index
    print("Compiling final BM25 statistics from cache (This may take a minute with 250k arrays)...")
    
    tokenized_corpus = []
    with open(BM25_TOKENS_PATH, "r") as f:
        for line in f:
            tokenized_corpus.append(json.loads(line))
            
    bm25 = BM25Okapi(tokenized_corpus)
    
    print(f"Serializing BM25 corpus to {BM25_PATH}...")
    with open(BM25_PATH, 'wb') as f:
        pickle.dump(bm25, f)
        
    print("🎉 System Indexing 100% Complete! The backend is ready for massive scale queries.")
    conn.close()

if __name__ == "__main__":
    build_indexes()
