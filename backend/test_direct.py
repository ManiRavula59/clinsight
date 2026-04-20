import os
from dotenv import load_dotenv
load_dotenv()
from twilio.rest import Client

# Credentials loaded from .env — never hardcode secrets in source files

try:
    client = Client(os.environ["TWILIO_ACCOUNT_SID"], os.environ["TWILIO_AUTH_TOKEN"])
    
    target_number = "+919493732359"
    webhook_url = "https://uyywn-157-50-156-32.run.pinggy-free.link/api/v1/twilio/voice"
    
    print(f"Initiating LIVE CONVERSATIONAL dial to {target_number}...")
    print(f"Routing live audio stream through Pinggy Webhook: {webhook_url}")
    
    call = client.calls.create(
        url=webhook_url,
        to=target_number,
        from_=os.environ["TWILIO_PHONE_NUMBER"]
    )
    
    print(f"✅ Twilio successfully triggered the cellular network! Call SID: {call.sid}")
    print("Your phone should be ringing natively in Telugu right now!")
    print("Pick it up and START TALKING! Gemini is listening!")
except Exception as e:
    print(f"❌ Twilio Network Diagnostics Failed: {str(e)}")
