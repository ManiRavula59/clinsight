import json
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from app.services.llm_manager import llm_manager

class KGCategory(BaseModel):
    category: str
    parents: List[str] = Field(description="Broader hierarchical classifications (e.g. STEMI -> Myocardial Infarction)")

class KGEntity(BaseModel):
    entity: str
    type: str = Field(description="Disease, Symptom, Drug, or Lab")
    umls_concept_id: str = Field(description="Estimated UMLS/SNOMED code if known, else N/A")
    synonyms: List[str]
    hierarchy: KGCategory

class KGContraindication(BaseModel):
    conflict: str
    reasoning: str
    severity: str = Field(description="High, Medium, Low")

class KnowledgeGraph(BaseModel):
    entities: List[KGEntity]
    contraindications: List[KGContraindication]
    expanded_search_query: str = Field(description="A massive string of original entities, hierarchical parents, and synonyms to feed into the FAISS Dense Retriever for mathematically perfect overlap.")

class KnowledgeGraphStructurer:
    """
    Deterministic Structured Logic Step (Pre-Retrieval).
    Takes raw parsed text, maps to ontology, builds hierarchy, and checks for contradictions.
    Solves the base paper's limitation by forcing structured logic BEFORE vector retrieval.
    """
    def __init__(self):
        self.llm = llm_manager.get_fallback_chain()

    def build_graph(self, clinical_text: str) -> Optional[KnowledgeGraph]:
        system_prompt = """You are a strictly deterministic Medical Knowledge Graph Engine.
Your job is to take unstructured clinical text and output a JSON Knowledge Graph.
For every medical entity found, you must:
1. Classify it (Disease, Symptom, Drug, Lab).
2. Estimate a UMLS/SNOMED code if possible.
3. Provide exact synonyms.
4. Provide the exact hierarchical parent (e.g. STEMI -> Myocardial Infarction).
5. Detect any contradictions/contraindications (e.g. bleeding + warfarin).

Finally, compile ALL synonyms, entities, and parents into a single `expanded_search_query` string. This string will be fed directly into a Dense Vector FAISS index to ensure mathematical precision during retrieval. Do not include unrelated text in the expanded query, only medical terms.

Return strictly in this JSON format:
{
  "entities": [{"entity": "...", "type": "...", "umls_concept_id": "...", "synonyms": ["..."], "hierarchy": {"category": "...", "parents": ["..."]}}],
  "contraindications": [{"conflict": "...", "reasoning": "...", "severity": "..."}],
  "expanded_search_query": "..."
}"""
        
        try:
            # We use an LLM configured for JSON mode if possible, but fallback chain usually returns raw text.
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Build a Knowledge Graph for this presentation: {clinical_text}")
            ]
            
            response = self.llm.invoke(messages)
            
            # Clean JSON if wrapped in markdown
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:-3]
            elif content.startswith("```"):
                content = content[3:-3]
                
            data = json.loads(content)
            return KnowledgeGraph(**data)
            
        except Exception as e:
            print(f"[KnowledgeGraph] Error building graph: {e}")
            return None

kg_structurer = KnowledgeGraphStructurer()
