from sentence_transformers import  SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
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
for index in top_indices:
    print("Document: ", documents[index])
    print("Score: ", similarities[0][index])
    print()




# best_index = similarities[0].argmax()
# print("Best match: ")
# print(documents[best_index])
#
# print("Similarity: ")
# print(similarities[0][best_index])

















