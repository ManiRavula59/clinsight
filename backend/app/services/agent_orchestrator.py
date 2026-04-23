from dotenv import load_dotenv
load_dotenv()

import json
from typing import TypedDict, Annotated, Sequence, Literal, Dict, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from app.services.llm_manager import llm_manager

# ──────────────────────────────────────────────────────────────────────────────
# STATE DEFINITION
# ──────────────────────────────────────────────────────────────────────────────
class ClinsightState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    intent: str  # "case_retrieval", "prescription_upload", "followup_setup", "chat", "patient_registration"
    extracted_prescription: Dict[str, Any]
    prescription_confirmed: bool

# ──────────────────────────────────────────────────────────────────────────────
# LLM SETUP
# ──────────────────────────────────────────────────────────────────────────────
llm = llm_manager.get_fallback_chain()

def _extract_text(content) -> str:
    """Safely extract a plain string from any Gemini content payload (str or list-of-dicts)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(item.get("text", ""))
            else:
                parts.append(str(item))
        return " ".join(parts).strip()
    return str(content)

# ──────────────────────────────────────────────────────────────────────────────
# NODES
# ──────────────────────────────────────────────────────────────────────────────

def intent_router_node(state: ClinsightState) -> Dict:
    """
    Passes through the intent that was already synchronously classified 
    in orchestrator_stream. This prevents the state from being overwritten 
    during follow-up scheduling.
    """
    return {"intent": state.get("intent", "chat")}


def extraction_node(state: ClinsightState) -> Dict:
    """
    Phase 1: Human-in-the-Loop Extraction.
    Uses LLM to extract structured data from the raw uploaded prescription string/image.
    """
    last_msg = _extract_text(state["messages"][-1].content)

    class PrescriptionExtraction(BaseModel):
        medications: list[str] = Field(description="Names of new medications being prescribed")
        dosages: list[str] = Field(description="Dosages for each medication (e.g., 500mg)")
        timings: list[str] = Field(description="When to take (e.g., 1x daily after food)")
        preferred_language: str = Field(description="Language for follow-up (English, Telugu, etc.)")
        patient_name: str = Field(description="Name of the patient", default="Unknown Patient")
        age: int | None = Field(description="Patient age in years if mentioned", default=None)
        weight_kg: float | None = Field(description="Patient weight in kilograms if mentioned (e.g. 70 kg)", default=None)
        allergies: list[str] = Field(description="Any allergies mentioned by the user", default=[])
        patient_history: list[str] = Field(description="Medical conditions or history mentioned", default=[])
        active_medications: list[str] = Field(description="Existing medications the patient is already taking", default=[])

    structured_llm = llm.with_structured_output(PrescriptionExtraction)

    extraction_system = (
        "You are a medical data extraction assistant. "
        "Extract the prescription details from the document provided. "
        "If the preferred language is not explicitly listed, default to English."
    )

    try:
        extracted_data = structured_llm.invoke([
            SystemMessage(content=extraction_system),
            HumanMessage(content=f"Extract prescription from: {last_msg}")
        ])
        data_dict = extracted_data.model_dump()

        confirmation_msg = AIMessage(content=f"""
**Prescription Extracted for Verification**

I have parsed the document. Please confirm if these details are perfectly accurate:
```json
{json.dumps(data_dict, indent=2)}
```
**Respond with:** `[Yes/Confirm]` to proceed to Safety Checks, or reply with edits.
""")

        return {
            "extracted_prescription": data_dict,
            "prescription_confirmed": False,
            "messages": [confirmation_msg]
        }

    except Exception as e:
        return {"messages": [AIMessage(content=f"Error parsing prescription: {str(e)}")]}

from app.services.safety_guardian import get_patient_profile, SafetyGuardian

def confirmation_checker_node(state: ClinsightState) -> Dict:
    """
    Checks if the doctor approved the extracted prescription.
    If approved, strictly runs the Safety Guardian checks immediately.
    """
    last_msg = _extract_text(state["messages"][-1].content).strip().lower()

    if last_msg in ["yes", "confirm", "looks good", "approved", "ok", "y", "yes proceed"]:
        rx_data = state.get("extracted_prescription", {})
        patient_name = rx_data.get("patient_name", "the patient")
        
        # Try to fetch from DB first
        db_profile = get_patient_by_name(patient_name)
        if db_profile:
            profile = db_profile
        else:
            # Fallback to dynamic profile from the extracted chat data
            profile = {
                "name": patient_name,
                "age": rx_data.get("age"),           # Pillar 6: Age-Based Dosing
                "weight_kg": rx_data.get("weight_kg"),  # Pillar 7: Weight-Based Dosing
                "allergies": [{"allergen": a} for a in rx_data.get("allergies", [])],
                "active_medications": [{"name": m} for m in rx_data.get("active_medications", [])],
                "lab_results": [{"test_name": "Condition", "value": h} for h in rx_data.get("patient_history", [])]
            }

        guardian = SafetyGuardian()
        report = guardian.validate_prescription(profile, rx_data)

        outcome = report.get("outcome", "BLOCKED" if not report.get("is_safe", True) else "SAFE")
        alerts_json = report.get("alerts", [])

        if outcome == "SAFE":
            msg = AIMessage(content=(
                f"✅ **Prescription Guardian: SAFE**\n\n"
                f"No conflicts found for **{profile['name']}**. Prescription is clinically appropriate.\n\n"
                f"*Would you like to schedule a Conversational Follow-Up call? (Reply 'setup follow up')*"
            ))
            return {"prescription_confirmed": True, "intent": "followup_setup", "messages": [msg]}

        elif outcome == "WARNING":
            alerts_text = "\n".join([
                f"- ⚠️ **{a.get('pillar', 'Note')}**: {a.get('description', '')} (Severity: {a.get('severity', 'Low')})"
                for a in alerts_json
            ])
            msg = AIMessage(content=(
                f"⚠️ **Prescription Guardian: WARNING**\n\n"
                f"Prescription for **{profile['name']}** can proceed, but please review:\n\n"
                f"{alerts_text}\n\n"
                f"**Guardian Reasoning:** {report.get('reasoning', '')}\n\n"
                f"*Prescription approved with caution. Reply 'setup follow up' to schedule a monitoring call.*"
            ))
            return {"prescription_confirmed": True, "intent": "followup_setup", "messages": [msg]}

        else:  # BLOCKED
            alerts_text = "\n".join([
                f"- 🚨 **{a.get('pillar', 'Issue')}**: {a.get('description', '')} (Severity: {a.get('severity', 'High')})"
                for a in alerts_json
            ])
            msg = AIMessage(content=(
                f"🚫 **PRE-PRESCRIPTION SAFETY INTERVENTION — BLOCKED**\n\n"
                f"**CRITICAL CONFLICTS FOR {profile['name'].upper()}:**\n\n"
                f"{alerts_text}\n\n"
                f"**Guardian Reasoning:** {report.get('reasoning', '')}\n\n"
                f"Prescription **BLOCKED**. Please revise and re-upload."
            ))
            return {"extracted_prescription": None, "prescription_confirmed": False, "intent": "chat", "messages": [msg]}
    else:
        return {
            "extracted_prescription": None,
            "prescription_confirmed": False,
            "intent": "chat",
            "messages": [AIMessage(content="Confirmation denied. Please upload the correct prescription or type the edits.")]
        }

import sqlite3

def get_patient_by_name(name: str):
    conn = sqlite3.connect("app/data/patients.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    row = cursor.execute("SELECT id FROM patients WHERE LOWER(name) LIKE ?", (f"%{name.lower()}%",)).fetchone()
    if row:
        from app.services.safety_guardian import get_patient_profile
        profile = get_patient_profile(row['id'])
        conn.close()
        return profile
    conn.close()
    return None

def case_retrieval_node(state: ClinsightState) -> Dict:
    # Stub - actual dispatch happens in orchestrator_stream directly to clinical_search_stream
    return {"messages": [AIMessage(content="[Routing to RAG Engine...]")]}

def patient_registration_node(state: ClinsightState) -> Dict:
    """Extracts a patient profile from chat and saves it to the SQLite DB."""
    last_msg = _extract_text(state["messages"][-1].content)

    class PatientRegistration(BaseModel):
        patient_name: str = Field(description="Name of the patient")
        age: int = Field(description="Age of the patient in years", default=0)
        weight_kg: float | None = Field(description="Patient weight in kilograms (e.g. 72.5 kg) if mentioned", default=None)
        gender: str = Field(description="Gender of the patient", default="Unknown")
        allergies: list[str] = Field(description="Any allergies mentioned", default=[])
        patient_history: list[str] = Field(description="Medical conditions or history", default=[])
        active_medications: list[str] = Field(description="Current medications", default=[])

    structured_llm = llm.with_structured_output(PatientRegistration)
    
    try:
        data = structured_llm.invoke([
            SystemMessage(content="Extract the patient profile details for database registration."),
            HumanMessage(content=last_msg)
        ])
        
        # Save to DB
        conn = sqlite3.connect("app/data/patients.db")
        cursor = conn.cursor()
        
        # Check if patient exists
        row = cursor.execute("SELECT id FROM patients WHERE LOWER(name) LIKE ?", (f"%{data.patient_name.lower()}%",)).fetchone()
        
        if row:
            patient_id = row[0]
            # Update Patient (including weight_kg)
            cursor.execute(
                "UPDATE patients SET age=?, weight_kg=?, chronic_conditions=? WHERE id=?",
                (data.age, data.weight_kg, ", ".join(data.patient_history), patient_id)
            )
            # Delete old records to replace them
            cursor.execute("DELETE FROM allergies WHERE patient_id=?", (patient_id,))
            cursor.execute("DELETE FROM active_medications WHERE patient_id=?", (patient_id,))
            cursor.execute("DELETE FROM lab_results WHERE patient_id=?", (patient_id,))
            msg_prefix = f"🔄 **Patient Profile Updated**\n\nPatient **{data.patient_name}** (ID: {patient_id}) already existed in the EMR. Their profile has been successfully updated with the latest details."
        else:
            # Insert Patient (including weight_kg)
            cursor.execute(
                "INSERT INTO patients (name, age, weight_kg, preferred_language, chronic_conditions) VALUES (?, ?, ?, ?, ?)",
                (data.patient_name, data.age, data.weight_kg, "English", ", ".join(data.patient_history))
            )
            patient_id = cursor.lastrowid
            msg_prefix = f"🏥 **Patient Registered Successfully**\n\nPatient **{data.patient_name}** (ID: {patient_id}) has been permanently saved to the Clinsight EMR Database."
        
        # Insert Allergies
        for allergy in data.allergies:
            cursor.execute("INSERT INTO allergies (patient_id, allergen, severity) VALUES (?, ?, ?)",
                           (patient_id, allergy, "High"))
                           
        # Insert Meds
        for med in data.active_medications:
            cursor.execute("INSERT INTO active_medications (patient_id, drug_name, dosage) VALUES (?, ?, ?)",
                           (patient_id, med, "Unknown"))
                           
        # Insert History as Labs for guardian check
        for hist in data.patient_history:
            cursor.execute("INSERT INTO lab_results (patient_id, test_name, value, unit, date) VALUES (?, ?, ?, ?, ?)",
                           (patient_id, "Condition", 0, hist, "2026-01-01"))
                           
        conn.commit()
        conn.close()
        
        msg = f"{msg_prefix}\n\nI have securely stored their allergies ({len(data.allergies)}), active medications ({len(data.active_medications)}), and medical history. You can now issue prescriptions for this patient by name, and I will automatically cross-reference their full profile."
        return {"intent": "chat", "messages": [AIMessage(content=msg)]}
        
    except Exception as e:
        return {"intent": "chat", "messages": [AIMessage(content=f"Error registering patient: {str(e)}")]}

from app.services.scheduler import schedule_call

def followup_setup_node(state: ClinsightState) -> Dict:
    """
    Schedules the voice conversational agent via APScheduler.
    """
    rx = state.get("extracted_prescription", {})
    if not rx:
        return {"messages": [AIMessage(content="No active prescription found in context to monitor. Upload a prescription first.")]}

    try:
        patient_name = rx.get("patient_name", "the patient")
        if patient_name == "Unknown Patient":
             patient_name = "Rajesh Kumar" # fallback if name wasn't provided
        
        schedule_call(patient_name=patient_name, language=rx.get("preferred_language", "Telugu"), rx_data=rx, delay_seconds=15)
        msg = AIMessage(content=f"📞 **Follow-up Scheduled Successfully**\n\nI have securely scheduled the Multi-Lingual Conversational Agent. It will awaken and dial {patient_name} exactly 15 seconds from now to track medication adherence.")
        return {"messages": [msg]}
    except Exception as e:
        return {"messages": [AIMessage(content=f"Scheduling error: {str(e)}")]}

def chat_node(state: ClinsightState) -> Dict:
    """
    Enhanced General Chat Agent to clarify doubts and answer questions.
    """
    sys_msg = SystemMessage(content=(
        "You are the Clinsight General Assistant. You help clinicians clarify doubts about the system, "
        "explain medical concepts in a general sense, and provide guidance on how to use Clinsight's "
        "Retrieval, Safety, and Voice agents. Be professional, concise, and helpful."
    ))
    # Use the LLM to generate a real response from the message history
    response = llm.invoke([sys_msg] + list(state["messages"]))
    return {"messages": [response]}

# ──────────────────────────────────────────────────────────────────────────────
# ROUTING LOGIC
# ──────────────────────────────────────────────────────────────────────────────
def route_intent(state: ClinsightState) -> str:
    intent = state.get("intent", "chat")
    if intent == "awaiting_confirmation":
        return "confirmation_checker_node"
    elif intent == "prescription_upload":
        return "extraction_node"
    elif intent == "case_retrieval":
        return "case_retrieval_node"
    elif intent == "followup_setup":
        return "followup_setup_node"
    elif intent == "patient_registration":
        return "patient_registration_node"
    else:
        return "chat_node"

# ──────────────────────────────────────────────────────────────────────────────
# GRAPH COMPILATION
# ──────────────────────────────────────────────────────────────────────────────
builder = StateGraph(ClinsightState)

builder.add_node("intent_router_node", intent_router_node)
builder.add_node("extraction_node", extraction_node)
builder.add_node("confirmation_checker_node", confirmation_checker_node)
builder.add_node("case_retrieval_node", case_retrieval_node)
builder.add_node("followup_setup_node", followup_setup_node)
builder.add_node("patient_registration_node", patient_registration_node)
builder.add_node("chat_node", chat_node)

builder.add_edge(START, "intent_router_node")

builder.add_conditional_edges(
    "intent_router_node",
    route_intent,
    {
        "confirmation_checker_node": "confirmation_checker_node",
        "extraction_node": "extraction_node",
        "case_retrieval_node": "case_retrieval_node",
        "followup_setup_node": "followup_setup_node",
        "patient_registration_node": "patient_registration_node",
        "chat_node": "chat_node"
    }
)

builder.add_edge("extraction_node", END)
builder.add_edge("confirmation_checker_node", END)
builder.add_edge("case_retrieval_node", END)
builder.add_edge("followup_setup_node", END)
builder.add_edge("patient_registration_node", END)
builder.add_edge("chat_node", END)

orchestrator_graph = builder.compile()

import asyncio
from app.services.omni_parser import ingest_document


def _quick_classify_intent(messages: list[dict]) -> str:
    """Fast synchronous intent classification using the LLM and chat history."""
    if len(messages) >= 2:
        last_asst = messages[-2].get('content', '')
        last_user = messages[-1].get('content', '').strip().lower()
        if "Prescription Extracted for Verification" in last_asst or "Please confirm" in last_asst:
            return "awaiting_confirmation"
            
    last_msg = messages[-1].get('content', '')
    router_system = (
        "You are the Clinsight Master Orchestrator routing clinical inputs.\n"
        "Classify the intent into EXACTLY ONE of: prescription_upload, case_retrieval, followup_setup, patient_registration, chat.\n"
        "case_retrieval: ANY clinical case, patient symptoms, disease, diagnosis queries, or finding similar patients. "
        "Examples: 'patient with chest pain', 'MI case', 'fever+cough', 'suspected pneumonia', '65yo man with dyspnea'.\n"
        "prescription_upload: User submitting medication/prescription data.\n"
        "patient_registration: User providing a detailed patient profile (name, age, history, allergies, meds) to save to the system.\n"
        "followup_setup: User scheduling a patient follow-up call.\n"
        "chat: ONLY pure non-clinical general conversation.\n"
        "When in doubt, choose case_retrieval. Reply with ONLY the exact string."
    )
    response = llm.invoke([
        SystemMessage(content=router_system),
        HumanMessage(content=last_msg)
    ])
    intent = _extract_text(response.content).strip().lower()
    valid_intents = ["prescription_upload", "case_retrieval", "followup_setup", "patient_registration", "chat"]
    return intent if intent in valid_intents else "chat"



async def orchestrator_stream(messages: list[dict]):
    """
    Smart SSE dispatcher:
      - case_retrieval → Full RAG pipeline (FAISS + ColBERT + Cross-Encoder + LLM racing)
                         → streams: trace, cases, chunk (LLM Summary / AI Thinking / Metrics), done
      - prescription_upload / followup_setup / chat → LangGraph orchestrator
    Manages base64 file intercepts via omni_parser.
    """
    from app.services.agent import clinical_search_stream

    last_ui_msg = messages[-1]
    file_data = last_ui_msg.get('file')

    # ── Document ingestion (prescription image / PDF) ───────────────────────
    if file_data:
        yield f"data: {json.dumps({'type': 'trace', 'content': 'Ingesting Omni-Channel Document (Vision/PDF Parser)...'})}\n\n"
        try:
            text = ingest_document(file_data['type'], file_data['base64'])
            last_ui_msg['content'] += f"\n\n[Extracted Document Data]:\n{text}"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': f'Omni-Parser Failed: {str(e)}'})}\n\n"
            return

    yield f"data: {json.dumps({'type': 'trace', 'content': 'Routing Intent through Master Agent...'})}\n\n"

    try:
        intent = _quick_classify_intent(messages)

        yield f"data: {json.dumps({'type': 'trace', 'content': f'Intent classified: {intent}'})}\n\n"

        # ── CASE RETRIEVAL: pipe directly into the full RAG pipeline ────────
        if intent == "case_retrieval":
            yield f"data: {json.dumps({'type': 'trace', 'content': 'Dispatching to Adaptive Case Retrieval Engine (FAISS + ColBERT + Cross-Encoder + LLM)...'})}\n\n"
            async for event in clinical_search_stream(messages):
                yield event
            return

        # ── ALL OTHER INTENTS: run through LangGraph orchestrator ────────────
        lc_messages = []
        for m in messages:
            if m['role'] == 'user':
                lc_messages.append(HumanMessage(content=m['content']))
            elif m['role'] == 'assistant':
                lc_messages.append(AIMessage(content=m['content']))

        # Reconstruct state from history if needed
        extracted_rx = None
        import re
        for m in reversed(messages):
            if m.get('role') == 'assistant':
                match = re.search(r'```json\n(.*?)\n```', m.get('content', ''), re.DOTALL)
                if match:
                    try:
                        extracted_rx = json.loads(match.group(1))
                        break
                    except Exception:
                        pass

        initial_state = {
            "messages": lc_messages,
            "intent": intent,
            "extracted_prescription": extracted_rx,
            "prescription_confirmed": False
        }
        
        final_state = orchestrator_graph.invoke(initial_state)
        final_msg = _extract_text(final_state["messages"][-1].content)
        handled_by = final_state.get('intent', 'unknown')
        yield f"data: {json.dumps({'type': 'trace', 'content': f'Routing Complete: Handled by {handled_by}'})}\n\n"

        words = final_msg.split(' ')
        for i, word in enumerate(words):
            suffix = ' ' if i < len(words) - 1 else ''
            yield f"data: {json.dumps({'type': 'chunk', 'content': word + suffix})}\n\n"
            await asyncio.sleep(0.01)

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
