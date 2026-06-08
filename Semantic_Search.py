from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

corpus = [
    "Python is a versatile programming language.",
    "Machine learning improves predictions with data.",
    "Web scraping extracts information from websites.",
    "Cybersecurity protects systems from attacks."
]

corpus_embeddings = model.encode(corpus, convert_to_tensor=True)

query = "How do I collect data from websites?"
query_embedding = model.encode(query, convert_to_tensor=True)

results = util.semantic_search(query_embedding, corpus_embeddings, top_k=2)

print("Top matches:")
for result in results[0]:
    print(f"- {corpus[result['corpus_id']]} (score: {result['score']:.4f})")
