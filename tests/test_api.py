"""
tests/test_api.py — tests básicos para ISO 45001 RAG API

Qué se testea:
  - GET /              → health check
  - GET /health        → estado del documento indexado
  - POST /ingest       → indexación de Markdown
  - POST /query/json   → respuesta RAG válida
"""

# ── GET / — health check ─────────────────────────────────────────────────────


def test_root_returns_200(client):
    resp = client.get("/")
    assert resp.status_code == 200


def test_root_has_required_keys(client):
    body = client.get("/").json()

    assert "status" in body
    assert "app" in body
    assert "model" in body
    assert "embed_model" in body
    assert "collection" in body
    assert "chunks_indexed" in body


def test_root_status_is_ok(client):
    body = client.get("/").json()
    assert body["status"] == "ok"


def test_root_chunks_indexed_is_int(client):
    body = client.get("/").json()

    assert isinstance(body["chunks_indexed"], int)
    assert body["chunks_indexed"] >= 0


def test_root_collection_is_iso(client):
    body = client.get("/").json()
    assert body["collection"] == "iso_45001_2018"


# ── GET /health ──────────────────────────────────────────────────────────────


def test_health_returns_200(client):
    resp = client.get("/health")
    assert resp.status_code == 200


def test_health_has_document_and_chunks(client):
    body = client.get("/health").json()

    assert "status" in body
    assert "document" in body
    assert "chunks_indexed" in body
    assert body["status"] == "healthy"


# ── POST /ingest — indexación Markdown ───────────────────────────────────────

SAMPLE_ISO_MD = """
# ISO 45001:2018

## 4 Contexto de la organización

La organización debe determinar las cuestiones internas y externas pertinentes
para su propósito y que afectan su capacidad para lograr los resultados previstos
del sistema de gestión de seguridad y salud en el trabajo.

## 5 Liderazgo y participación de los trabajadores

La alta dirección debe demostrar liderazgo y compromiso con respecto al sistema
de gestión de la SST.

## 6 Planificación

La organización debe determinar los riesgos y oportunidades que necesitan abordarse.
"""


def test_ingest_markdown_returns_200(client):
    resp = client.post(
        "/ingest",
        data={
            "text_content": SAMPLE_ISO_MD,
            "source_id": "iso_test_doc",
        },
    )

    assert resp.status_code == 200


def test_ingest_markdown_chunks_indexed_positive(client):
    resp = client.post(
        "/ingest",
        data={
            "text_content": SAMPLE_ISO_MD,
            "source_id": "iso_test_doc_chunks",
        },
    )

    body = resp.json()

    assert body["status"] == "ok"
    assert body["chunks_indexed"] >= 1


def test_ingest_markdown_collection_name(client):
    resp = client.post(
        "/ingest",
        data={
            "text_content": SAMPLE_ISO_MD,
            "source_id": "iso_test_doc_collection",
        },
    )

    body = resp.json()

    assert body["collection"] == "iso_45001_2018"


# ── POST /query/json — consulta RAG ──────────────────────────────────────────


def test_query_json_returns_200(populated_client):
    resp = populated_client.post(
        "/query/json",
        json={"question": "¿Cuál es el objetivo de la ISO 45001?"},
    )

    assert resp.status_code == 200


def test_query_json_has_answer(populated_client):
    body = populated_client.post(
        "/query/json",
        json={"question": "¿Qué debe demostrar la alta dirección?"},
    ).json()

    assert "answer" in body
    assert isinstance(body["answer"], str)
    assert len(body["answer"]) > 0


def test_query_json_has_sources(populated_client):
    body = populated_client.post(
        "/query/json",
        json={"question": "¿Qué son los riesgos y oportunidades?"},
    ).json()

    assert "sources" in body
    assert isinstance(body["sources"], list)


def test_query_json_has_confidence_note(populated_client):
    body = populated_client.post(
        "/query/json",
        json={"question": "¿Qué exige la norma sobre participación de trabajadores?"},
    ).json()

    assert "confidence_note" in body
    assert isinstance(body["confidence_note"], str)


# ── comportamiento con colección vacía ───────────────────────────────────────


def test_query_json_empty_collection_can_return_503(client):
    resp = client.post(
        "/query/json",
        json={"question": "¿Pregunta de prueba?"},
    )

    assert resp.status_code in (200, 503)

    if resp.status_code == 503:
        body = resp.json()
        assert "detail" in body
        assert len(body["detail"]) > 0
