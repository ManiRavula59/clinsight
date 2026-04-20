"""
bedrock_client.py
─────────────────
Provides the AWS Bedrock LLM client exclusively used by:
  - PatientGuardianVoiceAgent (Twilio voice calls)
  - WhatsApp Conversational Agent

Uses the Bearer API Key authentication (newer Bedrock API key format).
OpenRouter keys in llm_manager.py are NOT touched.
"""
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_aws import ChatBedrock
from langchain_openai import ChatOpenAI

_BEDROCK_API_KEY = os.getenv("AWS_BEDROCK_API_KEY", "")
_BEDROCK_REGION  = os.getenv("AWS_BEDROCK_REGION", "eu-north-1")

# Best Claude model available on Bedrock for conversational agents
_BEDROCK_MODEL = "anthropic.claude-3-5-haiku-20241022-v1:0"

def get_bedrock_llm():
    """
    Returns a LangChain ChatBedrock instance using the AWS Bedrock Bearer API Key.
    Falls back to OpenRouter if Bedrock key is missing.
    """
    if _BEDROCK_API_KEY:
        try:
            return ChatBedrock(
                model_id=_BEDROCK_MODEL,
                region_name=_BEDROCK_REGION,
                # Bedrock API key authentication (newer format)
                credentials_profile_name=None,
                model_kwargs={"max_tokens": 512, "temperature": 0.3},
                # Pass the bearer token via the endpoint config
                endpoint_url=f"https://bedrock-runtime.{_BEDROCK_REGION}.amazonaws.com",
                aws_access_key_id="BEDROCK_API_KEY",
                aws_secret_access_key=_BEDROCK_API_KEY,
            )
        except Exception as e:
            print(f"[Bedrock] Init failed, falling back to OpenRouter: {e}")

    # Graceful fallback to OpenRouter prescription key
    print("[Bedrock] API key missing or failed — using OpenRouter fallback for voice/WhatsApp")
    from app.services.llm_manager import llm_manager
    return llm_manager.get_fallback_chain()
