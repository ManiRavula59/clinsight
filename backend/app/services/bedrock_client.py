"""
bedrock_client.py
─────────────────
Custom LangChain chat model that calls AWS Bedrock REST API directly
using the new Bearer token (ABSK...) API key format.

This bypasses botocore/boto3 entirely — which only supports IAM credentials,
not the newer Bedrock API key format.

Model: anthropic.claude-3-5-haiku — fast, human-like, 200k token context.
Fallback: OpenRouter NVIDIA Nemotron if Bedrock fails.
"""

import os
import json
import requests
import logging
from typing import Any, Iterator, List, Optional

from dotenv import load_dotenv
load_dotenv()

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage, AIMessage, SystemMessage, HumanMessage
)
from langchain_core.outputs import ChatGeneration, ChatResult

logger = logging.getLogger("clinsight.bedrock")

_BEDROCK_API_KEY = os.getenv("AWS_BEDROCK_API_KEY", "")
_BEDROCK_REGION  = "us-east-1"   # Only region with API key access confirmed
_MODEL_ID        = "anthropic.claude-3-haiku-20240307-v1:0"  # ✅ Tested & working


def _to_anthropic_messages(messages: List[BaseMessage]):
    """Convert LangChain messages to Anthropic API format."""
    system_prompt = None
    anthropic_msgs = []
    for m in messages:
        if isinstance(m, SystemMessage):
            system_prompt = m.content
        elif isinstance(m, HumanMessage):
            anthropic_msgs.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage):
            anthropic_msgs.append({"role": "assistant", "content": m.content})
        else:
            anthropic_msgs.append({"role": "user", "content": str(m.content)})
    return system_prompt, anthropic_msgs


class BedrockBearerChat(BaseChatModel):
    """
    Direct REST call to AWS Bedrock using Bearer API key.
    Works with the new ABSK... API key format that botocore doesn't support.
    """
    api_key: str
    region: str = "eu-north-1"
    model_id: str = _MODEL_ID
    max_tokens: int = 4096
    temperature: float = 0.3

    @property
    def _llm_type(self) -> str:
        return "bedrock-bearer-direct"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager=None,
        **kwargs: Any,
    ) -> ChatResult:
        system_prompt, anthropic_msgs = _to_anthropic_messages(messages)

        payload: dict = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": anthropic_msgs,
        }
        if system_prompt:
            payload["system"] = system_prompt
        if stop:
            payload["stop_sequences"] = stop

        url = (
            f"https://bedrock-runtime.{self.region}.amazonaws.com"
            f"/model/{self.model_id}/invoke"
        )
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()

        data = response.json()
        text = data["content"][0]["text"]
        message = AIMessage(content=text)
        return ChatResult(generations=[ChatGeneration(message=message)])


def get_bedrock_llm():
    """
    Returns AWS Bedrock Claude (direct REST Bearer) with OpenRouter as automatic fallback.
    No startup probe — fallback is handled per-call by LangChain's with_fallbacks().
    """
    from app.services.llm_manager import llm_manager
    fallback_chain = llm_manager.get_fallback_chain()

    if _BEDROCK_API_KEY:
        bedrock = BedrockBearerChat(
            api_key=_BEDROCK_API_KEY,
            region=_BEDROCK_REGION,
            model_id=_MODEL_ID,
            max_tokens=2048,
            temperature=0.3,
        )
        logger.info("✅ AWS Bedrock Claude configured (Bearer REST) with OpenRouter fallback")
        return bedrock.with_fallbacks([fallback_chain])

    logger.info("🔄 AWS Bedrock key not found — using OpenRouter for voice/WhatsApp")
    return fallback_chain
