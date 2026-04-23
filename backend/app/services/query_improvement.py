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

    # Common English + medical stopwords to filter out of FTS5 queries
    STOPWORDS = {
        'a','an','the','is','was','were','are','be','been','being','have','has','had',
        'do','does','did','will','would','shall','should','may','might','must','can','could',
        'i','me','my','we','our','you','your','he','she','it','they','them','his','her','its',
        'this','that','these','those','am','at','by','for','from','in','into','of','on','to',
        'with','and','but','or','nor','not','no','so','if','then','than','too','very',
        'about','after','before','between','during','through','above','below','up','down',
        'out','off','over','under','again','further','once','here','there','when','where',
        'why','how','all','each','every','both','few','more','most','other','some','such',
        'only','own','same','just','also','now','patient','presented','admitted','history',
        'diagnosed','reported','showed','revealed','performed','noted','found','developed',
        'underwent','received','treated','case','year','old','male','female','man','woman',
        'day','days','month','months','week','weeks','hospital','department','clinical',
    }

    def _sanitize_for_fts5(self, text: str) -> str:
        """
        Extracts meaningful medical keywords for FTS5 OR-based search.
        Filters stopwords, keeps only clinical terms, uses OR logic.
        """
        # Strip non-alphanumeric
        cleaned = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        # Filter stopwords and short words
        words = [w for w in cleaned.lower().split() 
                 if w not in self.STOPWORDS and len(w) >= 3 and not w.isdigit()]
        # Take top 15 keywords max (prevents query explosion)
        keywords = words[:15]
        if not keywords:
            return cleaned.split()[0] if cleaned else "case"
        # Use OR logic so any keyword match counts
        return " OR ".join(keywords)

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
