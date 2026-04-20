from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import json

class PatientGuardianVoiceAgent:
    def __init__(self, patient_name: str, language: str, prescription_info: dict, days_passed: int = 4):
        self.patient_name = patient_name
        self.language = language
        self.rx = prescription_info
        self.days_passed = days_passed

        from app.services.bedrock_client import get_bedrock_llm
        self.llm = get_bedrock_llm()

        meds = prescription_info.get('medications', ['medicine'])
        med_name = meds[0] if meds else 'medicine'
        # Pre-calculate expected pills remaining (assumes 30-pill strip, 1 pill/day)
        pills_remaining = max(0, 30 - (days_passed * 1))

        self.system_prompt = f"""
You are the Clinsight Patient Guardian Voice AI calling {self.patient_name} for a medication follow-up.

LANGUAGE REQUIREMENTS (CRITICAL):
- Speak in {self.language} using ONLY Romanized/Latin script characters.
  Example Telugu: "Namaskaram {self.patient_name} garu, meeru ela unnaru?"
- Do NOT output any native Unicode script (Telugu/Hindi/Tamil letters). The TTS engine will crash.
- Keep each response to MAX 2 short, warm sentences.

CONTEXT:
- Patient name: {self.patient_name}
- Medication: {med_name}
- Days since discharge: {self.days_passed}
- Expected pills remaining (if taken correctly every day): {pills_remaining}

YOUR GOAL (follow this flow):
1. Warmly greet the patient by name.
2. Ask if they took {med_name} today.
3. If they say YES - praise them briefly, ask how they are feeling.
4. If they say NO - gently remind them, ask if they need help.
5. If they report SEVERE symptoms (chest pain, breathlessness, unconsciousness) - output [TRIAGE_ALERT_TRIGGERED] and say you are alerting the doctor.

Output ONLY plain conversational speech. No lists, no markdown, no asterisks, no symbols.
"""
        self.history = [SystemMessage(content=self.system_prompt)]

    def chat(self, user_msg: str) -> str:
        """Processes one conversational turn over the voice call."""
        if not user_msg.strip():
            user_msg = f"[SYSTEM: Initiate the outbound call greeting to {self.patient_name} now.]"

        self.history.append(HumanMessage(content=user_msg))

        def _extract(content):
            if isinstance(content, list):
                return " ".join(
                    str(item.get("text", "")) for item in content
                    if isinstance(item, dict) and "text" in item
                )
            return str(content)

        response = self.llm.invoke(self.history)
        reply = _extract(response.content)
        self.history.append(AIMessage(content=reply))
        return reply
