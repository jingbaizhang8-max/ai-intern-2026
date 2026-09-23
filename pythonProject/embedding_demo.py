from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

sentences = [
    "I love programming in Python.",
    "Python is my favorite programming language.",
    "I went swimming at the beach yesterday."
]

embeddings = model.encode(sentences)
print(type(embeddings))
print(embeddings.shape)
print(embeddings[0][:10])

similarity_1_2 = cosine_similarity(
    [embeddings[0]],
    [embeddings[1]]
)

similarity_1_3 = cosine_similarity(
    [embeddings[0]],
    [embeddings[2]]
)
print("Sentence 1 vs Sentence2: ", similarity_1_2[0][0])
print("Sentence 1 vs Sentence3: ", similarity_1_3[0][0])