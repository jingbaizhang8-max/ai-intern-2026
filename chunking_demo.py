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

chunks = chunk_text(text)

for i, chunk in enumerate(chunks):
    print(f"Chunk {i + 1}:")
    print(chunk)
    print()