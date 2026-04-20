import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

os.environ["GOOGLE_API_KEY"] = "AIzaSyBAM2MKqZ4_LHdApowPeUf1ZNpgdqUqppM"

def test_gemini():
    try:
        # test the exact alias string
        llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0)
        print("Model initialized.")
        res = llm.invoke([HumanMessage(content="Hello! Are you working?")])
        print("RESPONSE:", res.content)
    except Exception as e:
        print("ERROR:", str(e))

if __name__ == "__main__":
    test_gemini()
