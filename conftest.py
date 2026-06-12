"""
tests/conftest.py — Fixtures compartidas para toda la suite de tests

Sesión 3 · Bloque 2 (test_api.py) y Bloque 4 (test_contract, test_properties, test_evaluation)

PRINCIPIO CLAVE: todos los tests corren OFFLINE, sin GOOGLE_API_KEY real y sin
ChromaDB real. El objetivo es que `uv run pytest -q` funcione en CI (y en tu
laptop sin internet) sin costo ni flakiness de red.

Estrategia de mocking:
  1. Antes de importar la app, seteamos GOOGLE_API_KEY a un valor dummy para que
     el chequeo de startup no falle.
  2. Monkeypatcheamos `genai.Client` y las clases de chromadb para que la app
     inicialice contra stubs en memoria.
  3. El TestClient de FastAPI ejecuta el lifespan completo (startup/shutdown),
     así que los mocks deben estar en lugar antes de que app.py se importe.
"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# ── 1. Variable de entorno dummy ────────────────────────────────────────────
# Debe estar antes de cualquier import de la app para que la app no lance
# RuntimeError("GOOGLE_API_KEY no encontrada") durante el startup.
os.environ.setdefault("GOOGLE_API_KEY", "ci-dummy-key-no-llama-a-gemini")


# ── 2. Fakes para ChromaDB ──────────────────────────────────────────────────


class FakeCollection:
    """Colección ChromaDB en memoria — simula get_or_create_collection."""

    def __init__(self, name: str = "fake"):
        self.name = name
        self._docs: list[dict] = []

    def count(self) -> int:
        return len(self._docs)

    def upsert(self, ids, embeddings, documents, metadatas):
        for id_, doc, meta in zip(ids, documents, metadatas):
            self._docs.append({"id": id_, "document": doc, "metadata": meta})

    def query(self, query_embeddings, n_results, include, **kwargs):
        # Devuelve los primeros n_results documentos almacenados (o todos si hay menos)
        n = min(n_results, len(self._docs))
        docs = [d["document"] for d in self._docs[:n]]
        metas = [d["metadata"] for d in self._docs[:n]]
        dists = [0.1] * n
        return {
            "documents": [docs],
            "metadatas": [metas],
            "distances": [dists],
        }

    def add(self, ids, embeddings, documents, metadatas):
        self.upsert(ids, embeddings, documents, metadatas)


class FakeChromaClient:
    """Cliente ChromaDB en memoria."""

    def __init__(self, *args, **kwargs):
        self._collections: dict[str, FakeCollection] = {}

    def get_or_create_collection(self, name: str) -> FakeCollection:
        if name not in self._collections:
            self._collections[name] = FakeCollection(name)
        return self._collections[name]


# ── 3. Fake para Gemini ─────────────────────────────────────────────────────


def make_fake_genai_client():
    """Crea un cliente Gemini falso que devuelve respuestas predecibles."""
    client = MagicMock()

    # embed_content → devuelve un embedding sintético de 768 dimensiones
    fake_embedding = MagicMock()
    fake_embedding.values = [0.1] * 768
    embed_result = MagicMock()
    embed_result.embeddings = [fake_embedding]
    client.models.embed_content.return_value = embed_result

    fake_rag_response = json.dumps(
        {
            "answer": (
                "Según la ISO 45001:2018, la organización debe establecer, implementar, "
                "mantener y mejorar continuamente un sistema de gestión de seguridad y "
                "salud en el trabajo."
            ),
            "sources": ["iso_45001_2018 · ISO 45001:2018"],
            "confidence_note": "Basado en los fragmentos recuperados de la norma ISO 45001:2018.",
        }
    )
    gen_result = MagicMock()
    gen_result.text = fake_rag_response
    client.models.generate_content.return_value = gen_result

    return client


# ── 4. Fixture principal: TestClient con todo mockeado ──────────────────────


@pytest.fixture(scope="session")
def client():
    """
    TestClient de FastAPI con Gemini y ChromaDB mockeados.
    scope="session": se crea una vez por run de pytest (más rápido).

    Importante: el patch debe cubrir el módulo donde se USAN los símbolos
    (app.main), no donde se definen (chromadb, google.genai).
    """
    fake_chroma = FakeChromaClient()
    fake_genai = make_fake_genai_client()

    with (
        patch("app.main.chromadb.PersistentClient", return_value=fake_chroma),
        patch("app.main.chromadb.HttpClient", return_value=fake_chroma),
        patch("app.main.chromadb.CloudClient", return_value=fake_chroma),
        patch("app.main.genai.Client", return_value=fake_genai),
    ):
        # Importamos la app DESPUÉS de parchear para que el lifespan use los mocks
        from app.main import app

        with TestClient(app) as tc:
            yield tc


@pytest.fixture(scope="session")
def populated_client(client):
    """
    TestClient con al menos un documento ingestado.
    Necesario para que /query y /query/json no devuelvan 503.
    """
    # Ingestar un documento de texto sintético
    md_content = (
        "# ISO 45001:2018\n\n"
        "## Objetivo\n\n"
        "La ISO 45001 especifica requisitos para un sistema de gestión de seguridad "
        "y salud en el trabajo, con el objetivo de proporcionar lugares de trabajo "
        "seguros y saludables.\n\n"
        "## Liderazgo\n\n"
        "La alta dirección debe demostrar liderazgo y compromiso con el sistema "
        "de gestión de la SST.\n\n"
        "## Riesgos y oportunidades\n\n"
        "La organización debe determinar los riesgos y oportunidades que necesitan "
        "abordarse para lograr los resultados previstos del sistema de gestión.\n"
    )

    client.post(
        "/ingest",
        data={
            "text_content": md_content,
            "source_id": "iso_45001_2018",
        },
    )
    return client
