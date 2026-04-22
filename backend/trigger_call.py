import os
from twilio.rest import Client
from dotenv import load_dotenv

load_dotenv()

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")

# The user's phone number
TO_PHONE = "+919493732359"
# The cloudflare tunnel webhook url for the conversational agent
WEBHOOK_URL = "https://implies-first-jump-daily.trycloudflare.com/api/v1/twilio/voice"

def trigger_call():
    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    print(f"Triggering call from {TWILIO_PHONE_NUMBER} to {TO_PHONE}...")
    call = client.calls.create(
        to=TO_PHONE,
        from_=TWILIO_PHONE_NUMBER,
        url=WEBHOOK_URL,
        method="POST"
    )
    print(f"Call initiated! SID: {call.sid}")

if __name__ == "__main__":
    trigger_call()
