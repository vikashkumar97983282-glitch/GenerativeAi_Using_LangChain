from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding = HuggingFaceEmbeddings(model='sentence-transformers/all-MiniLM-L6-v2')

documents = [
    "Rohit Sharma is the captain of the Indian ODI team and is famous for his elegant batting and record-breaking double centuries.",
    "Virat Kohli is one of the greatest batsmen in cricket, known for his consistency, fitness, and aggressive style.",
    "Sachin Tendulkar is called the 'God of Cricket' and holds the record for 100 international centuries.",
    "MS Dhoni is India's legendary captain, known for his calm leadership, excellent wicketkeeping, and finishing ability.",
    "Jasprit Bumrah is one of the world's best fast bowlers, famous for his unique bowling action and deadly yorkers."
]


while True:

    query = input("Enter your query: ")

    if query.lower() == "exit":
            break


    doc_embeddings = embedding.embed_documents(documents)
    query_embedding = embedding.embed_query(query)

    scores = cosine_similarity([query_embedding], doc_embeddings)[0]

    index, score = sorted(list(enumerate(scores)),key=lambda x: x[1])[-1]


    print("query is :", query)
    print(documents[index])
    print("similarity score is :", score)

    



