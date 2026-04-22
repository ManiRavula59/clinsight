"""
Rebuild FAISS index on Colab GPU — upload pmc_cases.db to Drive first.
Downloads the finished index to Drive for you to copy back to your Mac.
"""

from google.colab import drive
drive.mount('/content/drive')

import subprocess
subprocess.check_call(["pip", "install", "-q", "faiss-cpu"])

import sqlite3, faiss, numpy as np, time, os
from sentence_transformers import SentenceTransformer

DB_PATH = "/content/drive/MyDrive/pmc_cases.db"
MODEL_PATH = "/content/drive/MyDrive/finetuned-pubmedbert"
INDEX_PATH = "/content/drive/MyDrive/disease_index.faiss"

print("=" * 60)
print("  FAISS Index Rebuild — Colab GPU")
print("=" * 60)

model = SentenceTransformer(MODEL_PATH, device="cuda")
dim = model.get_sentence_embedding_dimension()
print(f"[MODEL] {dim}d on CUDA")

conn = sqlite3.connect(DB_PATH)
rows = conn.execute("SELECT patient_uid, text FROM cases ORDER BY id").fetchall()
conn.close()
total = len(rows)
texts = [r[1][:800] for r in rows]
del rows
print(f"[DB] {total:,} cases")

print(f"[ENCODE] Encoding {total:,} cases on GPU...")
start = time.time()
embeddings = model.encode(
    texts,
    batch_size=512,
    show_progress_bar=True,
    normalize_embeddings=True,
    convert_to_numpy=True,
)
del texts
elapsed = time.time() - start
print(f"[ENCODE] Done in {elapsed/60:.1f} min ({total/elapsed:.0f} docs/s)")

index = faiss.IndexFlatIP(dim)
index.add(embeddings.astype(np.float32))
print(f"[FAISS] {index.ntotal:,} vectors")

faiss.write_index(index, INDEX_PATH)
print(f"[SAVED] {INDEX_PATH} ({os.path.getsize(INDEX_PATH)/1e6:.0f} MB)")
print("\n✅ Done! Download disease_index.faiss from Drive to backend/app/data/")
