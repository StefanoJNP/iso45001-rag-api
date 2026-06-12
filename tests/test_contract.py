"""
tests/test_contract.py — contract tests para ISO 45001 RAG API

Valida la forma de las respuestas.
Si cambia el contrato de la API, estos tests fallan.
"""

from pydantic import BaseModel


class RootContract(BaseModel):
    status: str
    app: str
    model: str
    embed_model: str
    collection: str
    chunks_indexed: int


class HealthContract(BaseModel):
    status: str
    document: str
    chunks_indexed: int


class IngestContract(BaseModel):
    status: str
    chunks_indexed: int
    collection: str


class RAGResponseContract(BaseModel):
    answer: str
    sources: list[str]
    confidence_note: str


# ── GET / ────────────────────────────────────────────────────────────────────


def test_root_contract_valid(client):
    resp = client.get("/")

    assert resp.status_code == 200

    contract = RootContract(**resp.json())

    assert contract.status == "ok"
    assert contract.collection == "iso_45001_2018"
    assert contract.chunks_indexed >= 0


def test_root_model_is_string(client):
    body = client.get("/").json()

    assert isinstance(body["model"], str)
    assert len(body["model"]) > 0


def test_root_chunks_indexed_non_negative(client):
    body = client.get("/").json()

    assert isinstance(body["chunks_indexed"], int)
    assert body["chunks_indexed"] >= 0


# ── GET /health ──────────────────────────────────────────────────────────────


def test_health_contract_valid(client):
    resp = client.get("/health")

    assert resp.status_code == 200

    contract = HealthContract(**resp.json())

    assert contract.status == "healthy"
    assert isinstance(contract.document, str)
    assert contract.chunks_indexed >= 0


# ── POST /ingest ─────────────────────────────────────────────────────────────

INGEST_PAYLOAD = {
    "text_content": """
# ISO 45001:2018

## Liderazgo

La alta dirección debe demostrar liderazgo y compromiso con el sistema de gestión
de seguridad y salud en el trabajo.
""",
    "source_id": "contract_iso_doc",
}


def test_ingest_contract_valid(client):
    resp = client.post("/ingest", data=INGEST_PAYLOAD)

    assert resp.status_code == 200

    contract = IngestContract(**resp.json())

    assert contract.status == "ok"
    assert contract.chunks_indexed >= 1
    assert contract.collection == "iso_45001_2018"


def test_ingest_chunks_indexed_is_int(client):
    resp = client.post("/ingest", data=INGEST_PAYLOAD)
    body = resp.json()

    assert isinstance(body["chunks_indexed"], int)


def test_ingest_collection_is_string(client):
    resp = client.post("/ingest", data=INGEST_PAYLOAD)
    body = resp.json()

    assert isinstance(body["collection"], str)
    assert len(body["collection"]) > 0


# ── POST /query/json ─────────────────────────────────────────────────────────


def test_query_contract_valid(populated_client):
    resp = populated_client.post(
        "/query/json",
        json={"question": "¿Cuál es el objetivo de la ISO 45001?"},
    )

    assert resp.status_code == 200

    contract = RAGResponseContract(**resp.json())

    assert len(contract.answer) > 0
    assert isinstance(contract.sources, list)
    assert isinstance(contract.confidence_note, str)


def test_query_answer_non_empty_string(populated_client):
    body = populated_client.post(
        "/query/json",
        json={"question": "¿Qué exige la norma sobre liderazgo?"},
    ).json()

    assert isinstance(body.get("answer"), str)
    assert len(body["answer"]) > 0


def test_query_sources_is_list(populated_client):
    body = populated_client.post(
        "/query/json",
        json={"question": "¿Qué son los riesgos y oportunidades?"},
    ).json()

    assert isinstance(body.get("sources"), list)


def test_query_confidence_note_exists(populated_client):
    body = populated_client.post(
        "/query/json",
        json={"question": "¿Qué indica la norma sobre mejora continua?"},
    ).json()

    assert "confidence_note" in body
    assert isinstance(body["confidence_note"], str)


# ── errores bien formados ────────────────────────────────────────────────────


def test_503_has_detail_field(client):
    resp = client.post(
        "/query/json",
        json={"question": "pregunta cualquiera"},
    )

    if resp.status_code == 503:
        body = resp.json()

        assert "detail" in body
        assert isinstance(body["detail"], str)
        assert len(body["detail"]) > 0


def test_ingest_without_content_returns_4xx(client):
    resp = client.post(
        "/ingest",
        data={"source_id": "empty_doc"},
    )

    assert resp.status_code in (400, 422)
