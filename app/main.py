"""
app/main.py — FastAPI demo para el curso GenAI Multimodal
Instructor: Rodrigo López Vera | Revolut Perú

Endpoints:
  GET  /           — health check (modelos activos + doc counts)
  POST /ingest     — recibe documento (texto o imagen), lo indexa en ChromaDB
  POST /query      — pregunta + imagen opcional → RAGResponse (multipart)
  POST /query/json — pregunta en JSON puro → RAGResponse (sin imagen)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CÓMO PROBAR — GUÍA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PASO 1 — Levanta la API
  export GOOGLE_API_KEY="A..." # pragma: allowlist secret
  uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

PASO 2 — Ingesta documentos (necesario antes del primer query)
-- Referir a taller pasado o:

  # Texto: circular SBS
  curl -X POST http://localhost:8000/ingest \\
       -F "file=@data/circulares_sbs/circular_B_2244_2024.md" \\
       -F "source_id=circular_B_2244_2024" \\
       -F "date=2024-03" \\
       -F "doc_type=text"

  # Imagen: voucher de pago
  curl -X POST http://localhost:8000/ingest \\
       -F "file=@data/images/voucher_yape_001.png" \\
       -F "source_id=voucher_yape_001" \\
       -F "date=2024-06" \\
       -F "doc_type=image"

PASO 3 — Consultas

  # Query simple (JSON)
  curl -X POST http://localhost:8000/query/json \\
       -H "Content-Type: application/json" \\
       -d '{"question": "¿Qué es una operación sospechosa?"}'

  # Query con filtro de fecha
  curl -X POST http://localhost:8000/query/json \\
       -H "Content-Type: application/json" \\
       -d '{"question": "Obligaciones del oficial de cumplimiento", "date_filter": "2024-01", "n_results": 5}'

  # Query multimodal (pregunta + voucher)
  curl -X POST http://localhost:8000/query \\
       -F "question=¿Esta transferencia requiere reporte a la UIF?" \\
       -F "image=@data/images/voucher_bbva_internacional_003.png"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CÓMO PROBAR DESDE PYTHON / GOOGLE COLAB
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  import httpx

  BASE = "http://localhost:8000"

  # 1. Health check
  print(httpx.get(f"{BASE}/").json())

  # 2. Ingestar circular SBS
  with open("data/circulares_sbs/circular_B_2244_2024.md", "rb") as f:
      r = httpx.post(f"{BASE}/ingest",
                     files={"file": f},
                     data={"source_id": "circular_B_2244_2024",
                           "date": "2024-03", "doc_type": "text"})
  print(r.json())  # {"status": "ok", "chunks_indexed": N, "collection": "circulares_sbs"}

  # 3. Ingestar voucher
  with open("data/images/voucher_yape_001.png", "rb") as f:
      r = httpx.post(f"{BASE}/ingest",
                     files={"file": f},
                     data={"source_id": "voucher_yape_001",
                           "date": "2024-06", "doc_type": "image"})
  print(r.json())  # {"status": "ok", "chunks_indexed": 1, "collection": "vouchers_financieros"}

  # 4. Query solo texto
  r = httpx.post(f"{BASE}/query/json",
                 json={"question": "¿Cuál es el umbral para reportar operaciones sospechosas?"})
  print(r.json())

  # 5. Query multimodal
  with open("data/images/voucher_bbva_internacional_003.png", "rb") as f:
      r = httpx.post(f"{BASE}/query",
                     data={"question": "¿Esta operación requiere reporte a la UIF?"},
                     files={"image": f}, timeout=30)
  print(r.json())

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SWAGGER UI — documentación interactiva en el navegador
  http://localhost:8000/docs
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

# Comentario test
import json
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import chromadb
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from google import genai
from google.genai import types
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite-preview")
EMBED_MODEL = os.environ.get("GEMINI_EMBED_MODEL", "gemini-embedding-2")

CHROMA_PATH = os.environ.get("CHROMA_PATH", "./chroma_db")
COLLECTION_NAME = "iso_45001_2018"

ISO_DOC_PATH = Path(
    os.environ.get(
        "ISO_DOC_PATH",
        "data/auditoria_sbs/iso_45001_2018.md",
    )
)

gemini_client = None
chroma_client = None
text_collection = None


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    question: str
    n_results: int = 3


class RAGResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence_note: str


class IngestResponse(BaseModel):
    status: str
    chunks_indexed: int
    collection: str


# ---------------------------------------------------------------------------
# Utilidades RAG
# ---------------------------------------------------------------------------


def split_markdown(md_text: str, max_chars: int = 2500) -> list[dict]:
    """
    Divide el Markdown en chunks por encabezados.
    Si un bloque es muy largo, lo parte por párrafos.
    """
    md_text = md_text.strip()

    sections = re.split(r"(?m)^(#{1,6}\s+.+)$", md_text)

    chunks = []

    if len(sections) <= 1:
        return split_large_text(md_text, max_chars=max_chars)

    current_heading = "Inicio"
    current_body = ""

    for part in sections:
        part = part.strip()

        if not part:
            continue

        if re.match(r"^#{1,6}\s+.+", part):
            if current_body.strip():
                chunks.extend(
                    build_chunks_from_section(
                        heading=current_heading,
                        body=current_body,
                        max_chars=max_chars,
                    )
                )

            current_heading = re.sub(r"^#{1,6}\s+", "", part).strip()
            current_body = ""
        else:
            current_body += "\n\n" + part

    if current_body.strip():
        chunks.extend(
            build_chunks_from_section(
                heading=current_heading,
                body=current_body,
                max_chars=max_chars,
            )
        )

    return chunks


def build_chunks_from_section(heading: str, body: str, max_chars: int) -> list[dict]:
    full_text = f"{heading}\n\n{body.strip()}"

    if len(full_text) <= max_chars:
        return [{"heading": heading, "text": full_text}]

    parts = split_large_text(full_text, max_chars=max_chars)

    for i, part in enumerate(parts, start=1):
        part["heading"] = f"{heading} - parte {i}"

    return parts


def split_large_text(text: str, max_chars: int = 2500) -> list[dict]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    chunks = []
    current = ""

    for paragraph in paragraphs:
        if len(current) + len(paragraph) + 2 <= max_chars:
            current = f"{current}\n\n{paragraph}".strip()
        else:
            if current:
                chunks.append({"heading": "Fragmento", "text": current})

            if len(paragraph) > max_chars:
                for i in range(0, len(paragraph), max_chars):
                    chunks.append(
                        {
                            "heading": "Fragmento largo",
                            "text": paragraph[i : i + max_chars],
                        }
                    )
                current = ""
            else:
                current = paragraph

    if current:
        chunks.append({"heading": "Fragmento", "text": current})

    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    embeddings = []

    for text in texts:
        result = gemini_client.models.embed_content(
            model=EMBED_MODEL,
            contents=text,
        )
        embeddings.append(result.embeddings[0].values)

    return embeddings


def ingest_markdown_file(path: Path) -> int:
    if not path.exists():
        raise RuntimeError(f"No se encontró el documento Markdown: {path}")

    md_text = path.read_text(encoding="utf-8")
    chunks = split_markdown(md_text)

    if not chunks:
        raise RuntimeError("El documento Markdown no generó chunks.")

    batch_size = 50

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [chunk["text"] for chunk in batch]
        embeddings = embed_texts(texts)

        ids = [f"iso_45001_2018__chunk_{i + j + 1}" for j in range(len(batch))]

        metadatas = [
            {
                "source": "iso_45001_2018",
                "heading": chunk["heading"],
                "doc_type": "markdown",
            }
            for chunk in batch
        ]

        text_collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    return len(chunks)


def retrieve_chunks(query: str, n_results: int = 3) -> list[dict]:
    if text_collection.count() == 0:
        raise HTTPException(
            status_code=503,
            detail="La colección está vacía. No se indexó la norma ISO 45001.",
        )

    query_embedding = embed_texts([query])[0]

    results = text_collection.query(
        query_embeddings=[query_embedding],
        n_results=min(n_results, text_collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    return [
        {
            "text": doc,
            "metadata": meta,
            "distance": distance,
        }
        for doc, meta, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]


def build_rag_response(question: str, chunks: list[dict]) -> RAGResponse:
    context = "\n\n---\n\n".join([f"[{chunk['metadata']['source']} | {chunk['metadata']['heading']}]\n{chunk['text']}" for chunk in chunks])

    sources = [f"{chunk['metadata']['source']} · {chunk['metadata']['heading']}" for chunk in chunks]

    prompt = f"""
Eres un asistente experto en auditoría, seguridad y salud en el trabajo e ISO 45001:2018.

Responde únicamente usando el contexto recuperado de la norma ISO 45001:2018.
Si la respuesta no está respaldada por el contexto, indícalo claramente.
Responde en español, de forma clara y útil para un auditor.

Devuelve la respuesta en JSON con esta estructura:
{{
  "answer": "...",
  "sources": ["..."],
  "confidence_note": "..."
}}

=== CONTEXTO ISO 45001:2018 ===
{context}

=== PREGUNTA ===
{question}
"""

    response = gemini_client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=700,
            response_mime_type="application/json",
        ),
    )

    try:
        parsed = json.loads(response.text)
    except json.JSONDecodeError:
        parsed = {
            "answer": response.text,
            "sources": sources,
            "confidence_note": "La respuesta fue generada con el contexto recuperado, pero no vino en JSON estricto.",
        }

    if not parsed.get("sources"):
        parsed["sources"] = sources

    if not parsed.get("confidence_note"):
        parsed["confidence_note"] = "Respuesta basada en los fragmentos recuperados de ISO 45001:2018."

    return RAGResponse(**parsed)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    global gemini_client, chroma_client, text_collection

    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY no encontrada. Pásala con -e GOOGLE_API_KEY=...")

    gemini_client = genai.Client(api_key=GOOGLE_API_KEY)

    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    text_collection = chroma_client.get_or_create_collection(COLLECTION_NAME)

    if text_collection.count() == 0:
        chunks_indexed = ingest_markdown_file(ISO_DOC_PATH)
        print(f"[startup] Documento ISO indexado: {chunks_indexed} chunks.")
    else:
        print(f"[startup] Chroma ya tiene {text_collection.count()} chunks.")

    yield

    print("[shutdown] Cerrando API.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ISO 45001 RAG API",
    description="API RAG para consultar una norma ISO 45001:2018 en Markdown",
    version="1.0.0",
    lifespan=lifespan,
)

Instrumentator().instrument(app).expose(app)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/")
async def root():
    return {
        "status": "ok",
        "app": "ISO 45001 RAG API",
        "model": MODEL,
        "embed_model": EMBED_MODEL,
        "collection": COLLECTION_NAME,
        "chunks_indexed": text_collection.count(),
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "document": str(ISO_DOC_PATH),
        "chunks_indexed": text_collection.count(),
    }


@app.post("/query/json", response_model=RAGResponse)
async def query_json(request: QueryRequest):
    chunks = retrieve_chunks(
        query=request.question,
        n_results=request.n_results,
    )

    return build_rag_response(
        question=request.question,
        chunks=chunks,
    )


@app.post("/query", response_model=RAGResponse)
async def query_form(
    question: str = Form(...),
    n_results: int = Form(default=3),
):
    chunks = retrieve_chunks(
        query=question,
        n_results=n_results,
    )

    return build_rag_response(
        question=question,
        chunks=chunks,
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    file: Optional[UploadFile] = File(default=None),
    text_content: Optional[str] = Form(default=None),
    source_id: str = Form(default="documento_adicional"),
):
    if file is not None:
        raw = await file.read()
        content = raw.decode("utf-8")
    elif text_content:
        content = text_content
    else:
        raise HTTPException(
            status_code=400,
            detail="Se requiere file o text_content.",
        )

    chunks = split_markdown(content)

    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="No se pudieron generar chunks del documento.",
        )

    batch_size = 50

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [chunk["text"] for chunk in batch]
        embeddings = embed_texts(texts)

        ids = [f"{source_id}__chunk_{i + j + 1}" for j in range(len(batch))]

        metadatas = [
            {
                "source": source_id,
                "heading": chunk["heading"],
                "doc_type": "markdown",
            }
            for chunk in batch
        ]

        text_collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    return IngestResponse(
        status="ok",
        chunks_indexed=len(chunks),
        collection=COLLECTION_NAME,
    )
