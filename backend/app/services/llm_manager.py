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
