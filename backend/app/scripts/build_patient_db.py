import sqlite3
import os

DB_PATH = "app/data/patients.db"

def init_db():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH) # Reset for clean state
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 1. Patients Table
    cursor.execute("""
    CREATE TABLE patients (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        age INTEGER,
        phone_number TEXT,
        preferred_language TEXT,
        chronic_conditions TEXT
    )
    """)

    # 2. Allergies Table
    cursor.execute("""
    CREATE TABLE allergies (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id INTEGER,
        allergen TEXT NOT NULL,
        severity TEXT,
        FOREIGN KEY(patient_id) REFERENCES patients(id)
    )
    """)

    # 3. Lab Results Table
    cursor.execute("""
    CREATE TABLE lab_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id INTEGER,
        test_name TEXT,
        value REAL,
        unit TEXT,
        date TEXT,
        FOREIGN KEY(patient_id) REFERENCES patients(id)
    )
    """)

    # 4. Active Medications
    cursor.execute("""
    CREATE TABLE active_medications (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id INTEGER,
        drug_name TEXT,
        dosage TEXT,
        FOREIGN KEY(patient_id) REFERENCES patients(id)
    )
    """)

    # --- MOCK DATA INSERTION --- #
    
    # Insert Rajesh (A complex patient to trigger safety checks)
    cursor.execute("""
        INSERT INTO patients (name, age, phone_number, preferred_language, chronic_conditions) 
        VALUES ('Rajesh Kumar', 55, '+919493732359', 'Telugu', 'Type 2 Diabetes, Hypertension')
    """)
    rajesh_id = cursor.lastrowid
    
    # Rajesh is allergic to Penicillin (Triggers Allergy Conflict)
    cursor.execute("INSERT INTO allergies (patient_id, allergen, severity) VALUES (?, ?, ?)", 
                   (rajesh_id, 'Penicillin', 'High - Anaphylaxis'))
                   
    # Rajesh has poor kidney function (Triggers Organ Function Threshold e.g. for high dose Metformin)
    cursor.execute("INSERT INTO lab_results (patient_id, test_name, value, unit, date) VALUES (?, ?, ?, ?, ?)", 
                   (rajesh_id, 'eGFR', 42.5, 'mL/min', '2026-03-01'))
                   
    # Rajesh is already taking a Blood Thinner (Triggers Drug-Drug interaction if Aminoglycoside/NSAID given)
    cursor.execute("INSERT INTO active_medications (patient_id, drug_name, dosage) VALUES (?, ?, ?)", 
                   (rajesh_id, 'Aspirin', '75mg daily'))

    # Insert Priya (A healthy patient)
    cursor.execute("""
        INSERT INTO patients (name, age, phone_number, preferred_language, chronic_conditions) 
        VALUES ('Priya Sharma', 28, '+918888888888', 'Hindi', 'None')
    """)

    conn.commit()
    conn.close()
    print("✅ Mock Patient Database constructed successfully at", DB_PATH)

if __name__ == "__main__":
    init_db()
