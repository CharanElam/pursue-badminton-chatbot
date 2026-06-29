# Pursue Badminton Academy Chatbot

A RAG (Retrieval-Augmented Generation) chatbot for Pursue Badminton Academy, built with LangChain, FAISS, HuggingFace embeddings, and Groq.

## Demo

Live app: [pursue-badminton-chatbot.streamlit.app](https://pursue-badminton-chatbot.streamlit.app)

## Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector store | FAISS |
| LLM | Groq (`llama-3.1-8b-instant`) |
| Orchestration | LangChain |

## How it works

1. Academy information is loaded from `data/academy_info.txt`
2. The text is split into chunks and embedded using a HuggingFace sentence-transformer model
3. Chunks are stored in a FAISS vector index
4. On each user query, the top-3 most relevant chunks are retrieved and injected into the LLM prompt
5. Groq returns a grounded answer based only on that context

## Local setup

**Prerequisites:** Python 3.10+

```bash
git clone https://github.com/CharanElam/pursue-badminton-chatbot.git
cd pursue-badminton-chatbot
pip install -r requirements.txt
```

Create `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "your-groq-api-key"
```

Get a free Groq API key at [console.groq.com](https://console.groq.com).

```bash
streamlit run app.py
```

## Deployment (Streamlit Community Cloud)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** → select your repo
3. Set **Main file path** to `app.py`
4. Under **Advanced settings → Secrets**, add:
   ```toml
   GROQ_API_KEY = "your-groq-api-key"
   ```
5. Click **Deploy**

## Project structure

```
├── app.py                  # Streamlit UI
├── rag.py                  # RAG chain (embeddings, retriever, LLM)
├── data/
│   └── academy_info.txt    # Academy knowledge base
├── requirements.txt
└── .streamlit/
    └── secrets.toml        # Local secrets (gitignored)
```

## Updating the knowledge base

Edit `data/academy_info.txt` with new information about the academy (schedule, pricing, coaches, etc.). The vector index is rebuilt on every cold start.
