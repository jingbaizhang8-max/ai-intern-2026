import os
from dotenv import load_dotenv
from  openai import  OpenAI
from sentence_transformers import SentenceTransformer
from  qdrant_client import QdrantClient, models


load_dotenv()
llm_client=OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)





documents = [
    {
        "text": """
        Employees receive 20 days of paid annual leave each year.
        Annual leave requests should normally be submitted at least
        two weeks before the planned leave date.
        """,
        "source": "employee_handbook.pdf",
        "page": 12
    },
    {
        "text": """
        Remote employees may work from home up to three days per week.
        Employees must receive approval from their manager before
        changing their regular remote work schedule.
        """,
        "source": "remote_work_policy.pdf",
        "page": 4
    },
    {
        "text": """
        Company laptops must use full-disk encryption.
        Employees must not share their passwords with other people.
        Lost devices should be reported to the IT department immediately.
        """,
        "source": "security_policy.pdf",
        "page": 7
    }
]


# for document in documents:
#     print("Source: ", document["source"])
#     print("Page: ",document["page"])
#     print("Text: ", document["text"])
#     print()

def chunk_text(text, chunk_size=12, overlap=3):
    words=text.split()
    chunks=[]
    start=0
    while start<len(words):
        end=min(start+chunk_size,len(words))
        chunk=" ".join(words[start:end])
        chunks.append(chunk)
        if end==len(words):
            break

        start=end-overlap
    return chunks

chunk_records=[]
for document in documents:
    chunks=chunk_text(document["text"])
    for chunk in chunks:
        chunk_record={
            "text": chunk,
            "source": document["source"],
            "page": document["page"]
        }
        chunk_records.append(chunk_record)

for i,record in enumerate(chunk_records):
    print(f"Chunk {i+1}")
    print("Text: ",record["text"])
    print("Source: ", record["source"])
    print("Page: ", record["page"])
    print()


model=SentenceTransformer(
"sentence-transformers/all-MiniLM-L6-v2"
)

qdrant_client=QdrantClient(":memory:")
qdrant_client.create_collection(
    collection_name="rag_citations",
    vectors_config=models.VectorParams(
        size=384,
        distance=models.Distance.COSINE
    )
)

texts=[]
for record in chunk_records:
    texts.append(record["text"])
embeddings=model.encode(texts)

points=[]
for i in range(len(chunk_records)):
    record=chunk_records[i]
    point=models.PointStruct(
        id=i,
        vector=embeddings[i].tolist(),
        payload={
            "text": record["text"],
            "source": record["source"],
            "page": record["page"]
        }
    )
    points.append(point)
    qdrant_client.upsert(
        collection_name="rag_citations",
        points=points
    )
    print("Stored chunks:", len(points))


query="How many days of annual leave do employees receive?"
query_embedding=model.encode(query)
results=qdrant_client.query_points(
    collection_name="rag_citations",
    query=query_embedding.tolist(),
    limit=3,
    with_payload=True
).points


# for i, result in enumerate(results):
#     print(f"\nResult {i + 1}")
#     print("Score:", result.score)
#     print("Text:", result.payload["text"])
#     print("Source:", result.payload["source"])
#     print("Page:", result.payload["page"])
#

retrieved_chunks=[]
for result in results:
    retrieved_chunks.append(result.payload["text"])
context="\n\n".join(retrieved_chunks)

sources=[]
for result in results:
    source_info = (
        result.payload["source"],
        result.payload["page"]
    )
    if source_info not in sources:
        sources.append(source_info)

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

response = llm_client.chat.completions.create(
    model="deepseek-flash",
    messages=messages
)

answer = response.choices[0].message.content


print("\nAnswer:")
print(answer)

print("\nSources:")

for source, page in sources:
    print(f"- {source} — page {page}")