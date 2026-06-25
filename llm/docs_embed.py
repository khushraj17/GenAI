from langchain_huggingface import HuggingFaceEmbeddings

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

vector = embedding.embed_documents(documents)

print(str(vector))