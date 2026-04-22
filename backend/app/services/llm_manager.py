import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel

OPENAI_API_BASE = "https://openrouter.ai/api/v1"
NVIDIA_MODEL    = "nvidia/nemotron-3-nano-30b-a3b:free"

# Keys 1-3: Synchronous pipeline (Planner, NER, KG, Safety Guardian, Orchestrator)
PRESCRIPTION_KEYS = [
    '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',
    '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',
    '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',
]

# Keys 4-6: Streaming pipeline (RAG LLM Synthesis)
RETRIEVAL_KEYS = [
    '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',
    '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',
    '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',
]

class LLMManager:
    """
    Two exclusive key pools to avoid exhausting all credits simultaneously.
    Pool A (PRESCRIPTION_KEYS 1-3): synchronous calls — Planner, NER, KG, Safety, Orchestrator.
    Pool B (RETRIEVAL_KEYS 4-6):   streaming calls   — RAG LLM Synthesis.
    """
    def __init__(self):
        self.prescription_clients = [
            ChatOpenAI(
                model=NVIDIA_MODEL,
                openai_api_base=OPENAI_API_BASE,
                openai_api_key=key,
                temperature=0,
                max_retries=0,
            ) for key in PRESCRIPTION_KEYS
        ]

        self.retrieval_clients = [
            ChatOpenAI(
                model=NVIDIA_MODEL,
                openai_api_base=OPENAI_API_BASE,
                openai_api_key=key,
                temperature=0.3,
                streaming=True,
                max_retries=0,
            ) for key in RETRIEVAL_KEYS
        ]

    def get_fallback_chain(self) -> BaseChatModel:
        """Sync chain: Key1 → Key2 → Key3."""
        primary  = self.prescription_clients[0]
        fallbacks = self.prescription_clients[1:]
        return primary.with_fallbacks(fallbacks) if fallbacks else primary

    def get_streaming_fallback_chain(self) -> BaseChatModel:
        """Streaming chain: Key4 → Key5 → Key6."""
        primary  = self.retrieval_clients[0]
        fallbacks = self.retrieval_clients[1:]
        return primary.with_fallbacks(fallbacks) if fallbacks else primary

# Singleton
llm_manager = LLMManager()
