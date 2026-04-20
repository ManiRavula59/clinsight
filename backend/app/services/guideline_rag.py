import json
from langchain_core.messages import SystemMessage, HumanMessage
from app.services.llm_manager import llm_manager

class GuidelineRAG:
    """
    Guideline RAG Branch.
    If the Clinical Planner routes here, we bypass patient DB retrieval and instead
    pull from general clinical guidelines (simulated here via LLM knowledge for prototype).
    """
    def __init__(self):
        self.llm = llm_manager.get_fallback_chain()

    async def get_guidelines(self, query: str) -> str:
        system_prompt = """You are a Clinical Guidelines RAG Engine.
Provide the current standard of care, protocols, or generic medical information regarding the user's query.
Format as clear, actionable clinical guidelines."""
        
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=query)]
        response = await self.llm.ainvoke(messages)
        return response.content

guideline_rag = GuidelineRAG()
