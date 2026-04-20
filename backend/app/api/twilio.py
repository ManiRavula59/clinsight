from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse
from app.services.patient_guardian_agent import PatientGuardianVoiceAgent
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import logging
import os
import json
logger = logging.getLogger("clinsight.twilio_webhook")
router = APIRouter(prefix="/api/v1/twilio", tags=["Twilio Voice Interaction"])

# Persistence layer for active calls to survive server reloads during debugging
HISTORY_DIR = "app/data/call_histories"
os.makedirs(HISTORY_DIR, exist_ok=True)

def save_history(call_sid, messages):
    path = f"{HISTORY_DIR}/{call_sid}.json"
    data = []
    for m in messages:
        if isinstance(m, SystemMessage): data.append({"role": "system", "content": m.content})
        elif isinstance(m, HumanMessage): data.append({"role": "human", "content": m.content})
        elif isinstance(m, AIMessage): data.append({"role": "ai", "content": m.content})
    with open(path, "w") as f:
        json.dump(data, f)

def load_history(call_sid):
    path = f"{HISTORY_DIR}/{call_sid}.json"
    if not os.path.exists(path): return None
    with open(path, "r") as f:
        data = json.load(f)
    messages = []
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
    for item in data:
        if item["role"] == "system": messages.append(SystemMessage(content=item["content"]))
        elif item["role"] == "human": messages.append(HumanMessage(content=item["content"]))
        elif item["role"] == "ai": messages.append(AIMessage(content=item["content"]))
    return messages

# In-memory store (secondary)
ACTIVE_CALLS = {}

@router.post("/voice")
async def handle_twilio_voice(
    request: Request,
    CallSid: str = Form(None), 
    SpeechResult: str = Form(None),
):
    """
    Twilio repeatedly hits this endpoint during a live phone call.
    """
    try:
        form_data = await request.form()
        logger.info(f"📥 RAW TWILIO FORM DATA: {dict(form_data)}")
        
        if not CallSid:
            logger.warning("⚠️ Received a webhook without CallSid. Likely a status check. Skipping logic.")
            return HTMLResponse(content="<Response></Response>", media_type="application/xml")
            
        logger.info(f"📞 Twilio Webhook Triggered for Call: {CallSid}")
        
        # 1. Initialize or Re-Hydrate the Agent
        rx_data = {"medications": ["Paracetamol 500mg"], "timings": ["Morning"]}
        agent = PatientGuardianVoiceAgent("Rajesh", "Telugu", rx_data)
        
        # Try to load history from disk if it exists (re-hydration)
        history = load_history(CallSid)
        if history:
            agent.history = history
            logger.info(f"🔄 Re-hydrated call history for {CallSid}")
        
        if not SpeechResult and not history:
            # Initial greeting case
            ai_reply = agent.chat("")
        elif SpeechResult:
            logger.info(f"🗣️ Transcribed Patient Audio: {SpeechResult}")
            ai_reply = agent.chat(SpeechResult)
        else:
            ai_reply = "Naku artham kaledu. Dayachesi malli cheppandi."

        # Save progress immediately
        save_history(CallSid, agent.history)
        
        logger.info(f"🤖 RAW AI REPLY: {ai_reply}")

        # 2. Check if the LLM threw the Medical Triage Alert flag
        is_triage = False
        if "[TRIAGE_ALERT_TRIGGERED]" in ai_reply:
            is_triage = True
            ai_reply = ai_reply.replace("[TRIAGE_ALERT_TRIGGERED]", "")
            ai_reply += " I am ending the call and alerting the doctor now. Dhanyavadalu."

        # Prevent 13521 Limits & 13520 Invalid Text Errors
        ai_reply = ai_reply.replace("<", "").replace(">", "").replace("&", "and").replace("*", "")
        if len(ai_reply) > 400:
            ai_reply = ai_reply[:400] + "... Please check with doctor."

        # 3. Construct the dynamic Twilio XML Response
        twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Pause length="1"/>
    <Say voice="Polly.Aditi" language="en-IN">{ai_reply}</Say>
"""
        
        if is_triage:
            twiml += "</Response>"
            logger.error("🚨 TRIAGE ALERT LOGGED TO DASHBOARD! Hanging up.")
        else:
            twiml += f"""
    <Gather input="speech" language="te-IN" speechTimeout="auto" action="/api/v1/twilio/voice">
    </Gather>
</Response>
"""

        # Cleanup whitespace for Twilio
        twiml = twiml.strip()
        logger.info(f"🎬 FINAL TWIML SENT: {twiml}")
        return HTMLResponse(content=twiml, media_type="application/xml")

    except Exception as e:
        logger.error(f"💥 CRITICAL WEBHOOK FAILURE: {str(e)}")
        # Fallback TwiML so the call doesn't just hang up with "Application Error"
        fallback = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Aditi" language="en-IN">Namaskaram, small technical issue. Please stay on the line.</Say>
    <Redirect>/api/v1/twilio/voice</Redirect>
</Response>""".strip()
        return HTMLResponse(content=fallback, media_type="application/xml")
