import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

os.environ["GOOGLE_API_KEY"] = "AIzaSyBAM2MKqZ4_LHdApowPeUf1ZNpgdqUqppM"

def test_gemini_lite():
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature=0)
        print("Lite Model initialized.")
        res = llm.invoke([HumanMessage(content="Test message")])
        print("RESPONSE:", res.content)
    except Exception as e:
        print("ERROR:", str(e))

if __name__ == "__main__":
    test_gemini_lite()
