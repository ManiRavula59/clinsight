import sqlite3
import json
from langchain_google_genai import ChatGoogleGenerativeAI
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
    return patient

from app.services.llm_manager import llm_manager

class SafetyGuardian:
    def __init__(self):
        self.llm = llm_manager.get_fallback_chain()
        
    def validate_prescription(self, patient_profile: dict, extracted_prescription: dict) -> dict:
        prompt = f"""
        You are the Pre-Prescription Safety Guardian.
        Check the following newly extracted prescription against the patient's existing profile.
        
        NEW PRESCRIPTION:
        {json.dumps(extracted_prescription, indent=2)}
        
        PATIENT EMR PROFILE:
        {json.dumps(patient_profile, indent=2)}
        
        Perform a strict 5-pillar check:
        1. Allergy Conflicts
        2. Drug-Drug Interactions (with active medications)
        3. Organ Function limits (check lab results, especially eGFR for renal clearance)
        4. Therapeutic Duplication
        5. QT Prolongation Risk (if applicable)
        
        Respond with a JSON object exactly matching this structure. 
        If any high/moderate severity issue is found, `is_safe` MUST be false.
        {{
            "is_safe": bool,
            "alerts": [
               {{"pillar": "Allergy", "description": "...", "severity": "High/Low"}}
            ],
            "reasoning": "Overall explanation"
        }}
        """
        try:
            res = self.llm.invoke(
                [
                    SystemMessage(content="You return strictly formatted JSON."), 
                    HumanMessage(content=prompt)
                ], 
                response_format={"type": "json_object"}
            )
            return json.loads(res.content)
        except Exception as e:
            return {
                "is_safe": False, 
                "alerts": [{"pillar": "System Error", "description": "Guardian engine failed.", "severity": "High"}], 
                "reasoning": str(e)
            }
