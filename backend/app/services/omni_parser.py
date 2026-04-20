import base64
import fitz  # PyMuPDF
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

def parse_pdf(file_path: str) -> str:
    """Extracts raw text from uploaded PDF prescriptions or labs."""
    text = ""
    doc = fitz.open(file_path)
    for page in doc:
        text += page.get_text()
    return text

def parse_image(base64_image: str) -> str:
    """Uses a Vision LLM (GPT-4o) to accurately transcribe handwritten or typed prescriptions."""
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", max_tokens=1500)
    msg = HumanMessage(
        content=[
            {
                "type": "text", 
                "text": "Extract all text, handwriting, and clinical details perfectly from this prescription image. Do not summarize or alter the meaning. Transcribe it directly."
            },
            {
                "type": "image_url", 
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            }
        ]
    )
    res = llm.invoke([msg])
    return res.content

def ingest_document(file_type: str, file_path_or_b64: str) -> str:
    """
    omni-channel router. Standardises any file into a raw text string
    that our LangGraph Orchestrator can ingest.
    """
    if file_type == "pdf" or file_type == "application/pdf":
        return parse_pdf(file_path_or_b64)
    elif file_type.startswith("image"):
        return parse_image(file_path_or_b64)
    
    return "[System: Unsupported file format]"
