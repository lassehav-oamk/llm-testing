import os
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# Initialize environment variable for Gemini API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set. Please set it.")

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

# Initialize sentence transformer for embeddings
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Sample knowledge base (in-memory list; could be loaded from a file)
knowledge_base = [
    "Vuoden 1991 painos MAOL-taulukoista on väriltään sini-punainen",
    "Nooran 7v lempiruoka on Hesburger.",
    "Hornet tippui osana esitystaitolentoharjoitusta toukokuussa 2025."
]

# Compute embeddings for the knowledge base
knowledge_embeddings = embedder.encode(knowledge_base)

# Compute cosine similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Retrieve relevant context
def retrieve_context(query, top_k=2):
    query_embedding = embedder.encode([query])[0]
    similarities = [cosine_similarity(query_embedding, emb) for emb in knowledge_embeddings]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    context = [knowledge_base[i] for i in top_indices]
    return "\n".join(context)

# Query Gemini API
def query_gemini(prompt):
    response = model.generate_content(prompt)
    return response.text.strip()

# Main RAG function
def rag_query(question):
    # Retrieve context
    context = retrieve_context(question)
    # Create prompt for Gemini
    prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
    # Get response from Gemini
    answer = query_gemini(prompt)
    return answer

# Example query
question = "Kerro eri vuosien MAOL-taulukoiden värit"
answer = rag_query(question)
print(f"Question: {question}")
print(f"Answer: {answer}")