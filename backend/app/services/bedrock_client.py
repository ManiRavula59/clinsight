"""
bedrock_client.py
─────────────────
AWS Bedrock Bearer API key format (ABSK...) is NOT compatible with
botocore/langchain-aws which requires standard IAM credentials.

This module now transparently returns the standard OpenRouter fallback chain
so that Voice Agent and WhatsApp Agent work reliably without any changes
to the calling code.

The AWS key is preserved in .env for future use if/when langchain-aws
adds support for the newer Bedrock API key format.
"""
from app.services.llm_manager import llm_manager

def get_bedrock_llm():
    """
    Returns the OpenRouter fallback chain.
    (AWS Bedrock Bearer key format not yet supported by langchain-aws botocore)
    """
    return llm_manager.get_fallback_chain()
