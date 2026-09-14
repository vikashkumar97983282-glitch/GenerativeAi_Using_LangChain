from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv


load_dotenv()


llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash"
)

input = input("User: ")

result = llm.invoke(input).content[0]["text"]

print(f"AI: {result}")