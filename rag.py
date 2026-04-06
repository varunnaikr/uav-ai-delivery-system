# rag.py

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')


# 📚 Load knowledge
def load_knowledge():
    with open("knowledge.txt", "r") as f:
        docs = f.readlines()
    return [doc.strip() for doc in docs]


# 🧠 Build FAISS index
def build_index(docs):
    embeddings = model.encode(docs)

    # Convert to float32 (IMPORTANT for FAISS)
    embeddings = np.array(embeddings).astype('float32')

    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return index, embeddings


# 🔍 Retrieve relevant knowledge
def retrieve(query, docs, index, top_k=3):
    query_vec = model.encode([query])
    query_vec = np.array(query_vec).astype('float32')

    distances, indices = index.search(query_vec, top_k)

    results = [docs[i] for i in indices[0]]
    return results