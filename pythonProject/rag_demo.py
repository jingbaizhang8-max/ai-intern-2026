import json
import os

from dotenv import load_dotenv
from  openai import OpenAI

from sentence_transformers import  SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")
client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"
)


model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)
documents = [
    "Python is a popular programming language used for AI and data science.",
    "FastAPI is a modern Python framework for building APIs.",
    "Docker packages applications into containers.",
    "RAG combines retrieval with large language models.",
    "Sydney is a city in Australia."
]

query = input("Ask a question: ")

document_embeddings = model.encode(documents)
query_embedding = model.encode([query])

similarities = cosine_similarity(
    query_embedding,
    document_embeddings
)

print(similarities)


top_k = 3
top_indices = similarities[0].argsort()[::-1][:top_k]

retrieved_documents = []
for index in top_indices:
    retrieved_documents.append(documents[index])

context = '\n'.join(retrieved_documents)

messages = [
    {
        "role": "system",
        "content": """
        You are a question-answering assistant.

        Answer the user's question using ONLY the provided context.
        
        If the answer cannot be found in the context,
        say: "I don't know based on the provided context."
        """
    },
    {
        "role": "user",
        "content": f"""
Context: {context}
Question: {query}
"""
    }
]



response = client.chat.completions.create(
    model="deepseek-flash",
    messages=messages
)

answer = response.choices[0].message.content

print("Answer: ")
print(answer)














