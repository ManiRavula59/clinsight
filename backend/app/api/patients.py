from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from app.services.safety_guardian import get_patient_profile, SafetyGuardian

router = APIRouter(prefix="/api/v1/patients", tags=["Patients"])
guardian = SafetyGuardian()

class PrescriptionRequest(BaseModel):
    extracted_prescription: Dict[str, Any]

@router.get("/{patient_id}")
async def fetch_patient(patient_id: int):
    """Fetches full structured EMR profile (allergies, labs, meds)"""
    profile = get_patient_profile(patient_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Patient not found")
    return profile

@router.post("/{patient_id}/validate-prescription")
async def validate_rx(patient_id: int, request: PrescriptionRequest):
    """
    Submits a JSON prescription payload to the Safety Guardian for
    7-pillar safety validation against the specific patient:
    1. Allergy Conflicts  2. Drug-Drug Interactions  3. Organ Function
    4. Therapeutic Duplication  5. QT Prolongation
    6. Age-Based Dosing (Beers Criteria for elderly, paediatric weight-calc flag)
    7. Weight-Based Dosing (mg/kg range check using patient weight_kg)
    """
    profile = get_patient_profile(patient_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Patient not found")
        
    safety_report = guardian.validate_prescription(profile, request.extracted_prescription)
    return safety_report
