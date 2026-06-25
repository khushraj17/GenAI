from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
load_dotenv()

embedding = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)

documents = [
"Artificial Intelligence enables machines to perform tasks that require human intelligence.",
"Machine learning algorithms improve their performance by learning from data.",
"Natural Language Processing helps computers understand and generate human language.",
"Deep learning uses neural networks with multiple layers to solve complex problems.",
"Computer vision allows machines to analyze and interpret images and videos."
]

query = "tell me about CV"

doc_embed = embedding.embed_documents(documents)
query_embed = embedding.embed_query(query)

scores = (cosine_similarity([query_embed],doc_embed))[0]


index, score = sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]

print(query)
print(documents[index])
print("similarity score is:", score)