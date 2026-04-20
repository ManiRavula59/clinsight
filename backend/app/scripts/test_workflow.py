import asyncio
import os
import json
from dotenv import load_dotenv
load_dotenv()

# Load credentials from .env — never hardcode secrets
os.environ.setdefault("TWILIO_ACCOUNT_SID", os.getenv("TWILIO_ACCOUNT_SID", ""))
os.environ.setdefault("TWILIO_AUTH_TOKEN", os.getenv("TWILIO_AUTH_TOKEN", ""))
os.environ.setdefault("TWILIO_PHONE_NUMBER", os.getenv("TWILIO_PHONE_NUMBER", ""))

from app.services.agent_orchestrator import orchestrator_stream
from app.services.scheduler import start_scheduler

async def run_scenario():
    print("==================================================")
    print("CLINSIGHT END-TO-END WORKFLOW INTEGRATION TEST")
    print("==================================================\\n")
    
    # We need to start the background caller first
    start_scheduler()

    # Step 1: Uploading the Rx to be parsed
    messages = [
        {"role": "user", "content": "I am prescribing Paracetamol 500mg once a day for a headache. Patient prefers Telugu followup."}
    ]
    
    print("\\n🚀 [STEP 1] DOCTOR UPLOADS PRESCRIPTION -> HITS INTENT ROUTER AND EXTRACTION NODE\\n")
    async for chunk in orchestrator_stream(messages):
        # Format the SSE chunk back into visible console log
        chunk_str = chunk.replace('data: ', '').strip()
        if chunk_str:
            try:
                data = json.loads(chunk_str)
                if data['type'] == 'trace':
                    print(f"   [TRACE] {data['content']}")
            except:
                pass


    # Simulated state insertion for the exact extraction parsing from step 1
    # Because our fake test script doesn't hook up a real database session to the UI, 
    # we simulate passing the state forward. In the real app, state is inferred by messages.
    
    print("\\n✅ [STEP 2] DOCTOR TYPES 'YES' -> TRIGGERS SAFETY GUARDIAN\\n")
    messages.append({
        "role": "assistant",
        "content": "**Prescription Extracted for Verification** Respond with: [Yes/Confirm] to proceed"
    })
    messages.append({
        "role": "user",
        "content": "yes"
    })
    
    # We inject the extracted payload into the last dictionary to simulate extraction node memory
    messages[-1]["extracted_prescription"] = {
        "medications": ["Paracetamol"],
        "dosages": ["500mg"],
        "timings": ["once a day"],
        "preferred_language": "Telugu"
    }

    async for chunk in orchestrator_stream(messages):
        chunk_str = chunk.replace('data: ', '').strip()
        if chunk_str:
            try:
                data = json.loads(chunk_str)
                if data['type'] == 'trace':
                    print(f"   [TRACE] {data['content']}")
            except:
                pass

    print("\\n✅ [STEP 3] DOCTOR SCHEDULES FOLLOWUP -> TRIGGERS APSCHEDULER EVENT\\n")
    messages.append({
        "role": "assistant",
        "content": "✅ **Prescription Guardian Passed** Would you like to schedule the Conversational Follow-Up Agent? (Reply 'setup follow up')"
    })
    messages.append({
        "role": "user",
        "content": "setup follow up"
    })
    
    # Keep context
    messages[-1]["extracted_prescription"] = messages[-2].get("extracted_prescription", {
        "medications": ["Paracetamol"],
        "preferred_language": "Telugu"
    })

    async for chunk in orchestrator_stream(messages):
        chunk_str = chunk.replace('data: ', '').strip()
        if chunk_str:
            try:
                data = json.loads(chunk_str)
                if data['type'] == 'chunk':
                    print(data['content'], end="")
            except:
                pass

    print("\n\n⏳ [STEP 4] WAITING 25 SECONDS FOR BACKGROUND AGENT TO WAKE UP AND DIAL...\n")
    for i in range(25):
        await asyncio.sleep(1)
        print(f"Waiting... {i+1}s")

if __name__ == "__main__":
    asyncio.run(run_scenario())
