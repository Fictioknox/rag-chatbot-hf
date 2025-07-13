import os
import gradio as gr
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import openai

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load documents
DOCS_DIR = "docs"
documents = []
filenames = []

for fname in os.listdir(DOCS_DIR):
    with open(os.path.join(DOCS_DIR, fname), "r", encoding="utf-8") as f:
        documents.append(f.read())
        filenames.append(fname)

# Create FAISS index
embeddings = model.encode(documents, convert_to_tensor=False)
index = faiss.IndexFlatL2(len(embeddings[0]))
index.add(np.array(embeddings).astype("float32"))

def retrieve(query, k=3):
    query_vec = model.encode([query])[0].astype("float32")
    D, I = index.search(np.array([query_vec]), k)
    return [documents[i] for i in I[0]]

def generate_answer(query):
    context = "\n".join(retrieve(query))
    prompt = f"Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response["choices"][0]["message"]["content"].strip()

def chatbot(query):
    try:
        return generate_answer(query)
    except Exception as e:
        return f"Error: {str(e)}"

demo = gr.Interface(fn=chatbot, inputs="text", outputs="text", title="RAG Q&A Chatbot")

if __name__ == "__main__":
    demo.launch()