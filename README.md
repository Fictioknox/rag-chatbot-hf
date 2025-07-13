# RAG Q&A Chatbot

This is a Retrieval-Augmented Generation (RAG) chatbot built using:
- FAISS for document retrieval
- Sentence-Transformers for embeddings
- OpenAI (gpt-3.5-turbo) for answer generation
- Gradio for a web UI

## 🌐 Live Demo
[Your Hugging Face Space URL will go here]

## 🛠 How to Use
1. Place your `.txt` files in the `docs/` folder
2. Add your OpenAI API key in Hugging Face secrets (see below)
3. Deploy to Hugging Face Spaces (Gradio template)

## 🔐 Hugging Face Secrets
Go to your Space > Settings > Repository secrets and add:
- `OPENAI_API_KEY`: Your OpenAI key

## 🧪 Example Queries
- What is artificial intelligence?
- What happened during the Renaissance?
- How does supply and demand work?