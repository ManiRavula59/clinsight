import os
from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel

# User's configured OpenRouter API Keys
OPENAI_API_BASE = "https://openrouter.ai/api/v1"
NVIDIA_MODEL = "nvidia/nemotron-3-nano-30b-a3b:free"

PRESCRIPTION_KEYS = [
    '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',
    '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',
    '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'
]

RETRIEVAL_KEYS = [
    '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',
    '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',
    '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'
]

class LLMManager:
    """
    Manages LLM instances using exclusive OpenRouter keys with the best NVIDIA model.
    Organizes keys into linear fallback chains to avoid exhausting all limits simultaneously.
    """
    def __init__(self):
        # We explicitly set max_retries=0 so that LangChain fails fast on Quota Hit
        
        # Prescription Engine (Synchronous) Pool
        self.prescription_clients = [
            ChatOpenAI(
                model=NVIDIA_MODEL,
                openai_api_base=OPENAI_API_BASE,
                openai_api_key=key,
                temperature=0,
                max_retries=0
            ) for key in PRESCRIPTION_KEYS
        ]
        
        # Retrieval (Streaming RAG) Pool
        self.retrieval_clients = [
            ChatOpenAI(
                model=NVIDIA_MODEL,
                openai_api_base=OPENAI_API_BASE,
                openai_api_key=key,
                temperature=0.3,
                streaming=True,
                max_retries=0
            ) for key in RETRIEVAL_KEYS
        ]

    def get_fallback_chain(self) -> BaseChatModel:
        """
        Returns a LangChain runnable for the Prescription Engine.
        Uses Key 1 -> Key 2 -> Key 3.
        """
        primary = self.prescription_clients[0]
        fallbacks = self.prescription_clients[1:]
        
        if fallbacks:
            return primary.with_fallbacks(fallbacks)
        return primary
        
    def get_streaming_fallback_chain(self) -> BaseChatModel:
        """
        Returns a LangChain runnable for the Streaming Case Retrieval Engine.
        Uses Key 4 -> Key 5 -> Key 6.
        """
        primary = self.retrieval_clients[0]
        fallbacks = self.retrieval_clients[1:]
        
        if fallbacks:
            return primary.with_fallbacks(fallbacks)
        return primary

# Singleton
llm_manager = LLMManager()
