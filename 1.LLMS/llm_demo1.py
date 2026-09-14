from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

texts = [
    "India is located in South Asia.",
    "New Delhi is the capital of India.",
    "Mumbai is the financial capital of India."
]

vectorstore = Chroma.from_texts(
    texts=texts,
    embedding=embeddings
)

results = vectorstore.similarity_search(
    "What is the capital of India?",
    k=2
)

for result in results:
    print(result.page_content)