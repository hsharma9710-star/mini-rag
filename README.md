# Mini RAG Backend (FastAPI + Supabase)

This project is a lightweight **Retrieval-Augmented Generation (RAG)** backend built using **FastAPI** and **Supabase**.

It supports:
- Text ingestion
- Vector storage
- Semantic search over stored documents

The system is deployed live using **Render**.

---

## 🚀 Live Demo

- **API Base URL:**  
  https://mini-rag-3was.onrender.com

- **Swagger Docs:**  
  https://mini-rag-3was.onrender.com/docs

---

## 🧠 How It Works

1. Text is ingested via `/ingest`
2. Text is split into chunks
3. Each chunk is converted into a fixed-size vector (384-dim lightweight embedding)
4. Vectors are stored in Supabase (Postgres + pgvector)
5. Queries are embedded and matched using cosine similarity

---

## 📌 API Endpoints

### `POST /ingest`
Stores text chunks and embeddings.

**Request**
```json
{
  "text": "RAG combines retrieval with generation."
}
