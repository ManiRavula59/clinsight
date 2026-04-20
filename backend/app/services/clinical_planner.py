import json
from typing import Dict, Any
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from app.services.llm_manager import llm_manager

class PlannerDecision(BaseModel):
    differential_diagnoses: list[str] = Field(description="List of potential differential diagnoses based on the query")
    required_evidence: list[str] = Field(description="Critical clinical data needed to confirm the diagnoses")
    search_refinements: str = Field(description="Suggested refined search query to retrieve the best cases")
    route: str = Field(description="Must be exactly 'retrieval' or 'guideline' or 'followup'")

class ClinicalPlanner:
    """
    Acts as the Clinical Planner LLM hypothesis generator.
    Forms differential diagnoses, identifies critical conditions,
    and decides the retrieval strategy.
    """
    def __init__(self):
        self.llm = llm_manager.get_fallback_chain()

    def route_query(self, query: str) -> PlannerDecision:
        system_prompt = """You are the Clinical Planner LLM.
Act as a clinical hypothesis generator. You must:
1. List differential diagnoses for the presentation.
2. Identify required evidence or missing critical conditions.
3. Suggest search refinements for the retrieval engine.
4. Decide if a guideline check is required (route: guideline) or if patient case retrieval is required (route: retrieval).

ROUTES:
- 'guideline': General medical guidelines, protocols, standards of care.
- 'retrieval': Patient presentation, symptoms, or specific case matching.
- 'followup': Conversational chit-chat.

Return strictly in JSON format:
{
  "differential_diagnoses": ["..."],
  "required_evidence": ["..."],
  "search_refinements": "...",
  "route": "retrieval|guideline|followup"
}"""
        try:
            messages = [SystemMessage(content=system_prompt), HumanMessage(content=query)]
            response = self.llm.invoke(messages)
            
            content = response.content.strip()
            if content.startswith("```json"): content = content[7:-3]
            elif content.startswith("```"): content = content[3:-3]
                
            data = json.loads(content)
            return PlannerDecision(**data)
        except Exception as e:
            print(f"[Clinical Planner] Error: {e}")
            return PlannerDecision(
                differential_diagnoses=[],
                required_evidence=[],
                search_refinements=query,
                route="retrieval"
            )

clinical_planner = ClinicalPlanner()
