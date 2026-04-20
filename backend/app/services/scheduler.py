from apscheduler.schedulers.asyncio import AsyncIOScheduler
import logging
import datetime
from app.services.patient_guardian_agent import PatientGuardianVoiceAgent

scheduler = AsyncIOScheduler()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("clinsight.scheduler")

def trigger_followup_call(patient_name: str, language: str, rx_data: dict, phone_number: str = "+919493732359"):
    """
    Background worker that hits Twilio to execute the call.
    """
    logger.info(f"\n\n====== ⏰ [VOICE AGENT WAKE-UP] ======")
    logger.info(f"Initiating outbound physical call to {patient_name} ({phone_number}) in {language}...")
    
    agent = PatientGuardianVoiceAgent(patient_name, language, rx_data)
    greeting = agent.chat("") # LLM Generates conversational opener
    
    # Send the LLM greeting to Twilio to strictly execute the physical phone call!
    from app.services.twilio_service import make_outbound_call
    
    lang_lower = language.lower()
    if "telugu" in lang_lower:
        twilio_lang = "te-IN"
    elif "tamil" in lang_lower:
        twilio_lang = "ta-IN"
    elif "hindi" in lang_lower:
        twilio_lang = "hi-IN"
    else:
        twilio_lang = "en-IN"
    success = make_outbound_call(phone_number, greeting, twilio_lang)
    
    if success:
        print(f"🎙️ [Call Successfully Placed to cell towers for {phone_number}]")
        logger.info(f"🎙️ [Call Successfully Placed to cell towers for {phone_number}]")
    else:
        print(f"❌ [Telephony Failed] Missing Twilio API details to execute physical ringing.")
        logger.error(f"❌ [Telephony Failed] Missing Twilio API details to execute physical ringing.")

    # Also send a WhatsApp follow-up message
    try:
        from app.api.whatsapp import send_whatsapp_message
        meds = rx_data.get('medications', ['your medicine'])
        med_name = meds[0] if meds else 'your medicine'
        wa_msg = (
            f"🏥 *Clinsight Medication Reminder*\n\n"
            f"Hello {patient_name}! This is your medication follow-up reminder.\n\n"
            f"💊 Have you taken your *{med_name}* today?\n"
            f"Reply *YES* if taken or *NO* if you missed it.\n\n"
            f"_Clinsight Medical Assistant_"
        )
        send_whatsapp_message(phone_number, wa_msg)
        logger.info(f"📱 WhatsApp follow-up sent to {phone_number}")
    except Exception as e:
        logger.warning(f"WhatsApp follow-up skipped: {e}")

def start_scheduler():
    scheduler.start()
    logger.info("⏱️ APScheduler Background Task Service Started. Ready for Medical Follow-up Assignments.")

def schedule_call(patient_name: str, language: str, rx_data: dict, delay_seconds: int = 15):
    """
    Schedules an asynchronous background outbound call.
    Usually this is set to 6:00 PM the next day, but for testing we use a delay.
    """
    run_date = datetime.datetime.now() + datetime.timedelta(seconds=delay_seconds)
    scheduler.add_job(
        trigger_followup_call, 
        'date', 
        run_date=run_date, 
        args=[patient_name, language, rx_data]
    )
    logger.info(f"✅ Call tracking locked. Agent scheduled to dial {patient_name} at {run_date}")
