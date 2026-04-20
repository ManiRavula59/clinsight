"""
migrate_similar_patients.py
============================
One-time migration: adds `similar_patients` column to pmc_cases.db
by reading from the PMC-Patients CSV file (which has the ground truth).

Run once:
    cd backend
    source venv/bin/activate
    python app/scripts/migrate_similar_patients.py
"""

import sqlite3
import csv
import ast
import time
import os
from huggingface_hub import hf_hub_download

DB_PATH = "app/data/pmc_cases.db"


def migrate():
    # ── 1. Download CSV ───────────────────────────────────────────────────
    print("Downloading PMC-Patients CSV from HuggingFace...")
    csv_path = hf_hub_download(
        repo_id="zhengyun21/PMC-Patients",
        filename="PMC-Patients.csv",
        repo_type="dataset",
    )
    print(f"CSV path: {csv_path}")

    # ── 2. Add column if it doesn't exist ─────────────────────────────────
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Check if column already exists
    c.execute("PRAGMA table_info(cases)")
    cols = [col[1] for col in c.fetchall()]
    if "similar_patients" not in cols:
        print("Adding 'similar_patients' column to cases table...")
        c.execute("ALTER TABLE cases ADD COLUMN similar_patients TEXT DEFAULT '{}'")
        conn.commit()
    else:
        print("Column 'similar_patients' already exists.")

    # ── 3. Build uid → similar_patients mapping from CSV ──────────────────
    print("Reading CSV to extract similar_patients ground truth...")
    start = time.time()
    uid_to_sp = {}
    total_csv = 0
    with_sp = 0

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_csv += 1
            uid = row.get("patient_uid", "").strip()
            sp_raw = row.get("similar_patients", "").strip()
            if uid and sp_raw and sp_raw != "{}":
                try:
                    sp_dict = ast.literal_eval(sp_raw)
                    if isinstance(sp_dict, dict) and len(sp_dict) > 0:
                        uid_to_sp[uid] = sp_dict
                        with_sp += 1
                except (ValueError, SyntaxError):
                    pass  # Skip malformed entries

    elapsed = time.time() - start
    print(f"CSV read in {elapsed:.1f}s: {total_csv:,} rows, {with_sp:,} have ground truth")

    # ── 4. Update SQLite using patient_uid as key ─────────────────────────
    print("Writing similar_patients to SQLite...")
    start = time.time()
    import json

    batch = []
    for uid, sp_dict in uid_to_sp.items():
        sp_json = json.dumps(sp_dict)
        batch.append((sp_json, uid))

    c.executemany("UPDATE cases SET similar_patients = ? WHERE patient_uid = ?", batch)
    conn.commit()
    elapsed = time.time() - start
    print(f"Updated {len(batch):,} rows in {elapsed:.1f}s")

    # ── 5. Verify ─────────────────────────────────────────────────────────
    c.execute(
        "SELECT COUNT(*) FROM cases WHERE similar_patients IS NOT NULL AND similar_patients != '{}'"
    )
    verified = c.fetchone()[0]
    print(f"\nVerification: {verified:,} cases now have ground truth in SQLite")

    # Show a sample
    c.execute(
        "SELECT id, patient_uid, similar_patients FROM cases WHERE similar_patients != '{}' LIMIT 2"
    )
    for row in c.fetchall():
        sp = json.loads(row[2])
        print(f"  ID={row[0]}, uid={row[1]}, similar_count={len(sp)}, first_3={dict(list(sp.items())[:3])}")

    conn.close()
    print("\nMigration complete!")


if __name__ == "__main__":
    migrate()
