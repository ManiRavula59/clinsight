import re
import json
from typing import Dict, Any, List
from pydantic import BaseModel
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from functools import lru_cache

load_dotenv()

from app.services.llm_manager import llm_manager

class Entity(BaseModel):
    text: str
    entity_type: str
    confidence: float
    negated: bool

class Concept(BaseModel):
    concept_id: str
    canonical_name: str
    original_mentions: List[str]
    semantic_type: str
    confidence: float

class QueryImprovementResult(BaseModel):
    normalized_text: str
    entities: List[Entity]
    concepts: List[Concept]
    expanded_terms: List[Dict[str, Any]]
    sparse_query: str = ""
    fts5_query: str = "" # Sanitized for SQLite FTS5 Match
    filters: Dict[str, Any] = {}

class QueryImprovementService:
    def __init__(self):
        # Uses primary keys first, openrouter last
        self.llm = llm_manager.get_fallback_chain()

    def _sanitize_for_fts5(self, text: str) -> str:
        """
        Final safety layer to prevent SQLite FTS5 syntax errors.
        Only allows alphanumeric characters and spaces.
        """
        # Replace non-alphanumeric with space
        cleaned = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        # Collapse multiple spaces
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        # Wrap each term in double quotes for exact keyword matching
        words = cleaned.split()
        return " ".join([f'"{w}"' for w in words])

    def _rule_normalize(self, text: str) -> str:
        """
        Local fallback to expand common medical shorthand when the LLM is rate-limited.
        """
        text_lower = " " + text.lower() + " "
        
        # Medical shorthand dictionary
        rules = {
            r'\by/o\b': 'year old',
            r'\byo\b': 'year old',
            r'\bmi\b': 'myocardial infarction',
            r'\bsob\b': 'shortness of breath',
            r'\bcabg\b': 'coronary artery bypass grafting',
            r'\bhtn\b': 'hypertension',
            r'\bdm\b': 'diabetes mellitus',
            r'\bcad\b': 'coronary artery disease',
            r'\badhd\b': 'attention deficit hyperactivity disorder',
            r'\bcopd\b': 'chronic obstructive pulmonary disease',
            r'\buti\b': 'urinary tract infection'
        }
        
        normalized = text_lower
        for pattern, expansion in rules.items():
            normalized = re.sub(pattern, expansion, normalized)
            
        return normalized.strip()

    @lru_cache(maxsize=256)
    def _process_cached(self, raw_text: str):
        """
        Inner cached method to prevent hitting LLM quota and
        eliminating 3+ seconds of latency on repeated requests.
        
        Uses LLM to:
        1. Expand clinical shortcuts (yo -> year old, MI -> myocardial infarction).
        2. Normalize medical grammar.
        3. Identify negated entities.
        4. Produce a sanitized string for FTS5.
        """
        
        prompt = (
            "You are a clinical informatics engine. Your task is to process a raw clinical query into a standardized form.\n"
            "1. Expand all clinical abbreviations (e.g., 'yo' to 'year old', 'MI' to 'myocardial infarction').\n"
            "2. Identify which clinical entities are NEGATED (e.g., 'no fever' means fever is negated).\n"
            "3. **CRITICAL**: Completely REMOVE the negated terms from the `normalized_text` and `keywords`. If the patient has 'no chest pain', do NOT include 'chest pain' in the normalized strings to prevent false-positive vector matching.\n"
            "Output ONLY a JSON object with this exact structure:\n"
            "{\n"
            "  \"normalized_text\": \"The positive symptoms only, expanded\",\n"
            "  \"entities\": [{\"text\": \"entity\", \"negated\": true/false, \"type\": \"symptom/diagnosis\"}],\n"
            "  \"keywords\": \"A clean space-separated string of the positive clinical keywords only\"\n"
            "}"
        )

        try:
            response = self.llm.invoke([
                SystemMessage(content=prompt),
                HumanMessage(content=f"Process this clinical query: {raw_text}")
            ])
            
            # Extract content
            content = response.content
            if isinstance(content, list):
                content = content[0].get("text", "") if isinstance(content[0], dict) else str(content[0])
            
            # Clean up potential markdown formatting in LLM response
            json_str = content.replace("```json", "").replace("```", "").strip()
            data = json.loads(json_str)
            
            normalized_text = data.get("normalized_text", raw_text)
            keywords = data.get("keywords", normalized_text)
            
            entities_data = data.get("entities", [])
            entities = [Entity(
                text=e["text"], 
                entity_type=e.get("type", "clinical"),
                confidence=1.0, 
                negated=e.get("negated", False)
            ) for e in entities_data]

            # Generate the FTS5-safe query from keywords
            fts5_query = self._sanitize_for_fts5(keywords)

            return QueryImprovementResult(
                normalized_text=normalized_text,
                entities=entities,
                concepts=[], # Concept linking can happen in next phase
                expanded_terms=[],
                sparse_query=keywords,
                fts5_query=fts5_query,
                filters={}
            )

        except Exception as e:
            print(f"Query Improvement LLM Error (Falling back to local rules): {e}")
            
            fallback_normalized = self._rule_normalize(raw_text)
            fallback_fts5 = self._sanitize_for_fts5(fallback_normalized)
            
            return QueryImprovementResult(
                normalized_text=fallback_normalized,
                entities=[], # We miss out on precise entity detection upon rate limit
                concepts=[],
                expanded_terms=[],
                sparse_query=fallback_normalized,
                fts5_query=fallback_fts5,
                filters={}
            )

    def process(self, raw_text: str, structured_fields: dict = None) -> QueryImprovementResult:
        """
        Uses LLM to:
        1. Expand clinical shortcuts.
        2. Normalize medical grammar.
        3. Produce a sanitized string for FTS5.
        (Wrapper to bypass unhashable dicts for lru_cache)
        """
        # Call the cached inner method
        result = self._process_cached(raw_text)
        
        # Inject the unhashable fields after retrieval
        result.filters = structured_fields or {}
        return result

# Singleton instance
query_improver = QueryImprovementService()
