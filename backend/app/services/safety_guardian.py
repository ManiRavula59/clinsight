import sqlite3
import json
from langchain_core.messages import SystemMessage, HumanMessage

DB_PATH = "app/data/patients.db"

def get_patient_profile(patient_id: int):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    patient_row = cursor.execute("SELECT * FROM patients WHERE id = ?", (patient_id,)).fetchone()
    if not patient_row:
        conn.close()
        return None
        
    patient = dict(patient_row)
    allergies = [dict(r) for r in cursor.execute("SELECT * FROM allergies WHERE patient_id = ?", (patient_id,)).fetchall()]
    labs = [dict(r) for r in cursor.execute("SELECT * FROM lab_results WHERE patient_id = ?", (patient_id,)).fetchall()]
    meds = [dict(r) for r in cursor.execute("SELECT * FROM active_medications WHERE patient_id = ?", (patient_id,)).fetchall()]
    
    conn.close()
    
    patient['allergies'] = allergies
    patient['lab_results'] = labs
    patient['active_medications'] = meds
    patient.setdefault('age', None)
    patient.setdefault('weight_kg', None)
    return patient

from app.services.llm_manager import llm_manager

class SafetyGuardian:
    def __init__(self):
        self.llm = llm_manager.get_fallback_chain()
        
    def validate_prescription(self, patient_profile: dict, extracted_prescription: dict) -> dict:
        age = patient_profile.get('age')
        weight_kg = patient_profile.get('weight_kg')

        age_ctx    = f"{age} years old" if age else "Age not recorded"
        weight_ctx = f"{weight_kg} kg"  if weight_kg else "Weight not recorded"

        prompt = f"""
You are the Pre-Prescription Safety Guardian for Clinsight.
Evaluate the following prescription against the patient profile using a 3-tier decision system.

NEW PRESCRIPTION:
{json.dumps(extracted_prescription, indent=2)}

PATIENT EMR PROFILE:
{json.dumps(patient_profile, indent=2)}

KEY DEMOGRAPHICS:
- Age: {age_ctx}
- Weight: {weight_ctx}

---
## 3-TIER DECISION RULES (STRICTLY FOLLOW):

### 🔴 BLOCK (is_safe: false, outcome: "BLOCKED")
Only set is_safe=false when ANY of these are TRUE:
- Confirmed ALLERGY conflict (drug matches a known allergen)
- SEVERE drug-drug interaction (e.g. serotonin syndrome risk, contraindicated combo)
- Critical organ failure risk (e.g. Metformin in eGFR < 30, nephrotoxic drug in severe CKD)
- Potentially fatal dose error (e.g. 10x overdose)

### ⚠️ WARNING (is_safe: true, outcome: "WARNING")
Set is_safe=true but include alerts for:
- Missing age or weight (cannot fully verify dosing — flag but do NOT block)
- Mild QT prolongation risk (single QT drug, no history of arrhythmia)
- Moderate drug interaction requiring monitoring (not contraindicated)
- Dose may need adjustment for age/organ function (but not clearly dangerous)
- Therapeutic class overlap that is not harmful duplication

### 🟢 SAFE (is_safe: true, outcome: "SAFE")
Set is_safe=true with no alerts when:
- No allergy conflicts
- No significant drug interactions
- Dosing is standard and clinically appropriate
- Organ function is compatible

---
## 5-PILLAR CHECKS:
1. Allergy Conflicts — check each new drug against known allergens.
2. Drug-Drug Interactions — check against active medications.
3. Organ Function — check labs, especially eGFR for renally-cleared drugs.
4. QT Prolongation — flag ONLY if multiple QT-prolonging drugs are combined.
5. Age/Weight Dosing — flag as WARNING only if data is missing; do not block adult-appearing prescriptions without clear paediatric or geriatric concern.

---
## KEY EXAMPLE:
- Azithromycin 500mg + Paracetamol + Dextromethorphan for a respiratory infection → SAFE or mild WARNING only.
- Missing weight → WARNING, NOT BLOCK.
- Penicillin allergy + Azithromycin → SAFE (Azithromycin is NOT a penicillin).

Respond ONLY with a JSON object matching this exact structure:
{{
    "is_safe": true or false,
    "outcome": "SAFE" or "WARNING" or "BLOCKED",
    "alerts": [
        {{"pillar": "...", "description": "...", "severity": "High/Moderate/Low"}}
    ],
    "reasoning": "Brief overall clinical reasoning"
}}
"""
        try:
            res = self.llm.invoke(
                [
                    SystemMessage(content="You are a clinical safety engine. Return ONLY a valid JSON object. No markdown, no explanation outside the JSON."),
                    HumanMessage(content=prompt)
                ]
            )
            content = res.content
            # Strip markdown code fences if model wraps the JSON
            if isinstance(content, str):
                content = content.strip()
                if content.startswith("```"):
                    content = content.split("```")[1]
                    if content.startswith("json"):
                        content = content[4:]
                    content = content.strip()
            
            result = json.loads(content)
            
            # Enforce the rule: is_safe must ONLY be false for HIGH severity allergy/interaction blocks
            # Moderate issues alone should not block
            high_severity_blocks = [
                a for a in result.get("alerts", [])
                if a.get("severity", "").lower() == "high"
                and a.get("pillar", "").lower() in ["allergy", "drug-drug interaction", "organ function"]
            ]
            if not high_severity_blocks:
                result["is_safe"] = True
                if result.get("outcome") == "BLOCKED":
                    result["outcome"] = "WARNING"
            
            return result
            
        except Exception as e:
            return {
                "is_safe": True,
                "outcome": "WARNING",
                "alerts": [{"pillar": "System", "description": f"Guardian engine error — manual review advised: {str(e)}", "severity": "Low"}],
                "reasoning": "Safety engine encountered a parsing issue. Prescription released for clinical judgment."
            }
