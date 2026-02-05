from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import tiktoken
import math
from supabase import create_client

load_dotenv()

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)


app = FastAPI()

class IngestRequest(BaseModel):
    text: str

class QueryRequest(BaseModel):
    query: str

def chunk_text(text: str, chunk_size=1000, overlap=120):
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)

    chunks = []
    start = 0

    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
        start += chunk_size - overlap

    return chunks

from openai import OpenAI

client = OpenAI()

def embed_text(chunks):
    embeddings = []
    for chunk in chunks:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk
        )
        embeddings.append(response.data[0].embedding)
    return embeddings


def cosine_similarity(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x*x for x in a))
    norm_b = math.sqrt(sum(x*x for x in b))
    return dot / (norm_a * norm_b)

@app.post("/ingest")
def ingest(req: IngestRequest):
    print("STEP 1: request received")

    chunks = chunk_text(req.text)
    print("STEP 2: chunks created", len(chunks))

    embeddings = embed_text(chunks)
    print("STEP 3: embeddings created", len(embeddings))

    rows = []
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        rows.append({
            "content": chunk,
            "embedding": emb,
            "metadata": {"source": "user_input", "chunk_index": i}
        })

    print("STEP 4: rows prepared")

    res = supabase.table("documents").insert(rows).execute()

    print("STEP 5: supabase insert result", res)

    return {"status": "ok", "rows": len(rows)}


@app.post("/query")
def query_docs(req: QueryRequest):
    query_embedding = embed_text([req.query])[0]
    data = supabase.table("documents").select("*").execute().data

    scored = []
    for row in data:
        embedding = row["embedding"]
        if isinstance(embedding, str):
            embedding = embedding.strip("[]")
            embedding = [float(x) for x in embedding.split(",")]

        score = cosine_similarity(query_embedding, embedding)
        scored.append((score, row))

    scored.sort(reverse=True, key=lambda x: x[0])
    top = scored[:3]

    if not top:
        return {
            "answer": "I don’t know based on the provided documents.",
            "citations": []
        }

    # Build answer from top chunks
    answer_parts = []
    citations = []

    for i, (score, row) in enumerate(top, start=1):
        answer_parts.append(f"[{i}] {row['content']}")
        citations.append({
            "id": i,
            "content": row["content"],
            "metadata": row["metadata"]
        })

    answer = (
        "Based on the retrieved documents, here is the answer:\n\n"
        + " ".join(answer_parts)
    )

    return {
        "answer": answer,
        "citations": citations
    }
