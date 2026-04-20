"""
Clinsight WhatsApp Conversational Agent
========================================
Works exactly like SBI/Jio WhatsApp chatbots.
Patients send WhatsApp messages → Twilio forwards to this webhook →
Our AI Agent responds → Twilio sends the reply back to the patient's WhatsApp.

Supports:
  - Patient: medication adherence follow-up conversations
  - Doctor: quick prescription safety checks via WhatsApp
  - Auto-language detection (Tamil, Telugu, Hindi, English)
"""

import os
import json
import logging
import sqlite3
from fastapi import APIRouter, Request, Form
from fastapi.responses import Response
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

logger = logging.getLogger("clinsight.whatsapp")
router = APIRouter(prefix="/api/v1/whatsapp", tags=["WhatsApp Agent"])

# Persistent session store: stores conversation history per WhatsApp number
WA_HISTORY_DIR = "app/data/whatsapp_sessions"
os.makedirs(WA_HISTORY_DIR, exist_ok=True)

# ─── Session Persistence ───────────────────────────────────────────────────────

def _session_path(wa_number: str) -> str:
    safe = wa_number.replace("+", "").replace(":", "_")
    return f"{WA_HISTORY_DIR}/{safe}.json"

def load_session(wa_number: str) -> list:
    path = _session_path(wa_number)
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        raw = json.load(f)
    msgs = []
    for item in raw:
        role = item.get("role")
        content = item.get("content", "")
        if role == "system":
            msgs.append(SystemMessage(content=content))
        elif role == "human":
            msgs.append(HumanMessage(content=content))
        elif role == "ai":
            msgs.append(AIMessage(content=content))
    return msgs

def save_session(wa_number: str, messages: list):
    path = _session_path(wa_number)
    data = []
    for m in messages:
        if isinstance(m, SystemMessage):
            data.append({"role": "system", "content": m.content})
        elif isinstance(m, HumanMessage):
            data.append({"role": "human", "content": m.content})
        elif isinstance(m, AIMessage):
            data.append({"role": "ai", "content": m.content})
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def clear_session(wa_number: str):
    path = _session_path(wa_number)
    if os.path.exists(path):
        os.remove(path)

# ─── DB Helpers ────────────────────────────────────────────────────────────────

def get_patient_by_phone(phone: str) -> dict | None:
    """Look up patient by their WhatsApp phone number."""
    try:
        conn = sqlite3.connect("app/data/patients.db")
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        # Strip whatsapp: prefix added by Twilio e.g. "whatsapp:+919493732359"
        clean_phone = phone.replace("whatsapp:", "").strip()
        row = cursor.execute(
            "SELECT id FROM patients WHERE phone_number = ? OR phone_number = ?",
            (clean_phone, phone)
        ).fetchone()
        if row:
            from app.services.safety_guardian import get_patient_profile
            profile = get_patient_profile(row["id"])
            conn.close()
            return profile
        conn.close()
    except Exception as e:
        logger.error(f"DB lookup error: {e}")
    return None

# ─── Language Detection ────────────────────────────────────────────────────────

def detect_language(text: str) -> str:
    """Simple Unicode-range detection for Indian languages."""
    for char in text:
        cp = ord(char)
        if 0x0C00 <= cp <= 0x0C7F:
            return "Telugu"
        if 0x0B80 <= cp <= 0x0BFF:
            return "Tamil"
        if 0x0900 <= cp <= 0x097F:
            return "Hindi"
    return "English"

# ─── TwiML Response Builder ────────────────────────────────────────────────────

def wa_reply(message: str) -> Response:
    """Returns a TwiML XML response that Twilio sends as a WhatsApp message."""
    # Escape XML special characters
    safe_msg = (message
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;"))
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Message>{safe_msg}</Message>
</Response>"""
    return Response(content=twiml, media_type="application/xml")

# ─── Main WhatsApp Webhook ─────────────────────────────────────────────────────

@router.post("/webhook")
async def whatsapp_webhook(
    request: Request,
    From: str = Form(None),
    Body: str = Form(None),
    ProfileName: str = Form(None),
):
    """
    Twilio sends every incoming WhatsApp message here.
    From: "whatsapp:+919493732359"
    Body: The actual text the patient/doctor typed
    ProfileName: Their WhatsApp display name
    """
    form_data = await request.form()
    logger.info(f"📱 WhatsApp Incoming | From: {From} | Name: {ProfileName} | Body: {Body}")

    if not From or not Body:
        return wa_reply("Hello! Send us a message to get started with Clinsight.")

    body = Body.strip()
    sender_name = ProfileName or "User"
    language = detect_language(body)

    # ── RESET command ──────────────────────────────────────────────────────────
    if body.lower() in ["reset", "new", "start", "hi", "hello", "hii", "namaste"]:
        clear_session(From)
        return wa_reply(
            f"👋 Welcome to *Clinsight Medical Assistant*, {sender_name}!\n\n"
            "I can help you with:\n"
            "1️⃣ *Medication reminders* — Are you a patient tracking your medicines?\n"
            "2️⃣ *Prescription safety check* — Are you a doctor checking a prescription?\n"
            "3️⃣ *Case retrieval* — Find similar medical cases\n\n"
            "Just type your query and I'll assist you! 🏥"
        )

    # ── Load conversation history ──────────────────────────────────────────────
    history = load_session(From)

    # ── First message — build a smart system prompt ────────────────────────────
    if not history:
        patient_profile = get_patient_by_phone(From)
        if patient_profile:
            patient_ctx = (
                f"Patient name: {patient_profile.get('name', sender_name)}\n"
                f"Allergies: {[a.get('allergen') for a in patient_profile.get('allergies', [])]}\n"
                f"Active medications: {[m.get('drug_name') for m in patient_profile.get('active_medications', [])]}\n"
                f"Conditions: {patient_profile.get('chronic_conditions', 'None')}"
            )
        else:
            patient_ctx = f"Name: {sender_name} (not yet registered in EMR)"

        system_prompt = (
            f"You are the Clinsight WhatsApp Medical Assistant, a concise and caring AI.\n"
            f"You are having a WhatsApp conversation with {sender_name}.\n"
            f"Detected language: {language}. Reply in the SAME language as the user.\n"
            f"For Tamil/Telugu/Hindi: use Romanized transliteration (NOT native script).\n"
            f"Keep responses SHORT (max 3 sentences) — this is WhatsApp, not a report.\n"
            f"Use emojis sparingly to be warm and professional.\n\n"
            f"Patient context from EMR:\n{patient_ctx}\n\n"
            f"Capabilities:\n"
            f"- Medication adherence check\n"
            f"- Prescription safety (allergy + drug interaction)\n"
            f"- Clinical case retrieval (symptoms → similar cases)\n"
            f"- General medical Q&A\n\n"
            f"IMPORTANT: If a prescription safety check is needed, flag conflicts clearly with ⚠️."
        )
        history = [SystemMessage(content=system_prompt)]

    # ── Add user message and get AI response ───────────────────────────────────
    history.append(HumanMessage(content=body))

    try:
        from app.services.bedrock_client import get_bedrock_llm
        llm = get_bedrock_llm()
        response = llm.invoke(history)

        # Extract text content safely
        content = response.content
        if isinstance(content, list):
            content = " ".join(
                item.get("text", "") for item in content if isinstance(item, dict)
            )
        ai_reply = str(content).strip()

        history.append(AIMessage(content=ai_reply))
        save_session(From, history)

        logger.info(f"📤 WhatsApp Reply to {From}: {ai_reply[:80]}...")
        return wa_reply(ai_reply)

    except Exception as e:
        logger.error(f"WhatsApp agent error: {e}")
        return wa_reply("⚠️ Sorry, our medical AI is temporarily busy. Please try again in a moment.")


# ─── Proactive WhatsApp Message Sender ────────────────────────────────────────

def send_whatsapp_message(to_number: str, message: str) -> bool:
    """
    Sends a proactive WhatsApp message to a patient.
    Used by the scheduler for medication follow-up reminders.
    e.g., send_whatsapp_message("+919493732359", "Good morning! Have you taken your Metformin today?")
    """
    twilio_sid = os.getenv("TWILIO_ACCOUNT_SID", "")
    twilio_token = os.getenv("TWILIO_AUTH_TOKEN", "")
    twilio_whatsapp_number = os.getenv("TWILIO_WHATSAPP_NUMBER", "whatsapp:+14155238886")  # Twilio sandbox default

    if not twilio_sid or not twilio_token:
        logger.error("❌ Twilio credentials missing. Cannot send WhatsApp message.")
        return False

    try:
        from twilio.rest import Client
        client = Client(twilio_sid, twilio_token)
        msg = client.messages.create(
            from_=twilio_whatsapp_number,
            to=f"whatsapp:{to_number}",
            body=message
        )
        logger.info(f"✅ WhatsApp message sent! SID: {msg.sid}")
        return True
    except Exception as e:
        logger.error(f"❌ WhatsApp send failed: {e}")
        return False
