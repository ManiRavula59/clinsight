from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
import json

@tool
def calculate_pill_count(total_pills_in_strip: int, days_passed: int, pills_prescribed_per_day: int) -> int:
    """Calculates the exact number of pills that SHOULD be remaining in the strip."""
    return total_pills_in_strip - (days_passed * pills_prescribed_per_day)

class PatientGuardianVoiceAgent:
    def __init__(self, patient_name: str, language: str, prescription_info: dict, days_passed: int = 4):
        self.patient_name = patient_name
        self.language = language
        self.rx = prescription_info
        self.days_passed = days_passed
        
        from app.services.llm_manager import llm_manager
        self.llm = llm_manager.get_fallback_chain()
        self.llm_with_tools = self.llm.bind_tools([calculate_pill_count])
        
        self.system_prompt = f"""
        You are the Clinsight Patient Guardian Voice AI. 
        You are speaking to your patient, {self.patient_name}, over a phone call.
        
        LANGUAGE REQUIREMENTS (CRITICAL): 
        You MUST speak fluently in {self.language}, but you MUST output the text entirely in Romanized/Transliterated English characters (e.g., Namaskaram, meeru ela unnaru). 
        Do NOT use native alphabet scripts. The TTS engine will crash if you output native Unicode.
        
        FORMAT: Maximum 1 short sentence! Use only English letters. NO TELUGU SCRIPT.
        
        Context: Discharge {self.days_passed} days ago. Prescription: {json.dumps(self.rx)}
        
        GOAL: Verify {self.rx.get('medications', ['medicine'])[0]} adherence.
        1. Greet briefly.
        2. Ask: "Tablet veskunnara?"
        3. If Yes/No, ask for pill count.
        
        Brief, Romanized, Warm. Output only plain text.
        """
        
        self.history = [SystemMessage(content=self.system_prompt)]
        
    def chat(self, user_msg: str) -> str:
        """Processes a turn of conversation, optionally executing tools if needed."""
        if not user_msg.strip():
            user_msg = f"Initiate the first outbound phone call greeting sequence directly natively to {self.patient_name}."
            
        self.history.append(HumanMessage(content=user_msg))
        
        def _extract_string(content):
            if isinstance(content, list):
                return " ".join(str(item.get("text", "")) for item in content if isinstance(item, dict) and "text" in item)
            return str(content)

        response = self.llm_with_tools.invoke(self.history)
        
        # Automatic Tool Execution Router
        if hasattr(response, "tool_calls") and response.tool_calls:
            self.history.append(response) # Append the AIMessage that requested the tool
            for tcall in response.tool_calls:
                if tcall["name"] == "calculate_pill_count":
                    result = calculate_pill_count.invoke(tcall["args"])
                    self.history.append(SystemMessage(content=f"Math check: {result} pills should be left."))
            
            final_response = self.llm_with_tools.invoke(self.history)
            self.history.append(final_response)
            return _extract_string(final_response.content)
            
        self.history.append(response)
        return _extract_string(response.content)
