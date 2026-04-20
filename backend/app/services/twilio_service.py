import os
import logging
from twilio.rest import Client

logger = logging.getLogger("clinsight.twilio")

# Provide your Twilio keys here or in .env
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER", "")

def make_outbound_call(to_phone_number: str, message_to_speak: str, language_code: str = "en-IN"):
    """
    Actually rings a physical phone number via Twilio.
    The language_code must map to Twilio's acceptable Polly/Neural text-to-speech formats.
    Telugu: te-IN, Hindi: hi-IN, Tamil: ta-IN
    """
    if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
        logger.error("🚫 TWILIO CREDENTIALS MISSING. Physical phone call cannot be dialed.")
        return False
        
    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    
    # We construct TwiML dynamically to use Twilio's Neural TTS
    # For a real Conversational AI, you would map this to a server Webhook (ngrok or live URL) 
    # that streams <Connect><Stream url="wss://..."/></Connect> to OpenAI's Realtime API!
    
    # Map to explicit Twilio Polly Voices to prevent 13512 Basic Engine crashes
    if language_code == "te-IN":
        voice = "Polly.Shruti"
    elif language_code == "hi-IN":
        voice = "Polly.Aditi"
    else:
        voice = "Polly.Joanna"
        
    twiml_payload = f"""<Response>
        <Pause length="2"/>
        <Say voice="{voice}" language="{language_code}">{message_to_speak}</Say>
    </Response>"""
    
    try:
        call = client.calls.create(
            twiml=twiml_payload,
            to=to_phone_number,
            from_=TWILIO_PHONE_NUMBER
        )
        print(f"📞 Twilio outbound call triggered! SID: {call.sid}")
        logger.info(f"📞 Twilio outbound call triggered! SID: {call.sid}")
        return True
    except Exception as e:
        print(f"❌ Twilio routing failed: {str(e)}")
        logger.error(f"❌ Twilio routing failed: {str(e)}")
        return False
