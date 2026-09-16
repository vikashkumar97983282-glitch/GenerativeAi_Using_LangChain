from langchain_huggingface import HuggingFaceEmbeddings


embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')


# text = "Patna is the capital of Bihar"

# vector = embedding.embed_query(text)


documents = [
    "patna is the capital of bihar",
    "bihar is  a state in india",
    "patna is a city in bihar"
]

vector = embedding.embed_documents(documents)


print(str(vector))