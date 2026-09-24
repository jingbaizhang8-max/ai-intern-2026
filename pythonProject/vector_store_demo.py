import  os
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from qdrant_client import  QdrantClient, models

load_dotenv()
llm_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)


model = SentenceTransformer(
"sentence-transformers/all-MiniLM-L6-v2"
)

qdrant_client = QdrantClient(":memory:")

qdrant_client.create_collection(
    collection_name="rag_chunks",
    vectors_config=models.VectorParams(
        size=384,
        distance=models.Distance.COSINE
    )
)

def chunk_text(text, chunk_size=12, overlap=3):
    words = text.split()
    chunks = []
    start = 0
    while start< len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        if end== len(words):
            break

        start = end - overlap

    return chunks

text = """
Retrieval augmented generation combines information retrieval
with large language models. Before answering a question,
the system searches for relevant information from a knowledge base.
The retrieved information is then provided to the language model
as context. Vector databases are commonly used to store embeddings
and perform similarity search.
"""

chunks= chunk_text(text)

embeddings = model.encode(chunks)

points = []

for i in range(len(chunks)):
    point = models.PointStruct(
        id = i,
        vector=embeddings[i].tolist(),
        payload={
            "text": chunks[i]
        }
    )
    points.append(point)
qdrant_client.upsert(
    collection_name="rag_chunks",
    points = points
)

print("Stored chunks", len(points))


query = "What are vector databases used for?"
query_embedding = model.encode(query)
results= qdrant_client.query_points(
    collection_name="rag_chunks",
    query=query_embedding.tolist(),
    limit=3,
    with_payload=True
).points

threshold = 0.3
if not results or results[0].score < threshold:
    print("\nFinal Answer:")
    print("I don't know based on the provided context.")
else:
    retrieved_chunks = []
    for result in results:
        retrieved_chunks.append(result.payload["text"])
    context = "\n\n".join(retrieved_chunks)

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
    Context:
    {context}
    
    Question:
    {query}
    """
        }
    ]

    response=llm_client.chat.completions.create(
        model="deepseek-flash",
        messages=messages
    )
    answer=response.choices[0].message.content

    print("\nFinal Answer: ")
    print(answer)



for i,result in enumerate(results):
    print(f"\nResult {i+1}")
    print("Score: ", result.score)
    print("Text: ", result.payload["text"])





