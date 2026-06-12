"""
tests/test_properties.py — property-based tests para ISO 45001 RAG API

Invariantes:
  1. La API no debe retornar 500 para preguntas válidas.
  2. Si retorna 200, answer debe ser string no vacío.
  3. Si retorna 200, sources debe ser una lista.
  4. Después de ingestar texto válido, chunks_indexed debe ser >= 0.
  5. GET / siempre debe estar disponible.
"""

from hypothesis import given, settings
from hypothesis import strategies as st


@settings(max_examples=30, deadline=10_000)
@given(question=st.text(min_size=1, max_size=500))
def test_query_never_returns_500(populated_client, question):
    resp = populated_client.post(
        "/query/json",
        json={"question": question},
    )

    assert resp.status_code in (200, 503), f"La API retornó {resp.status_code} para la pregunta: {question!r}\nBody: {resp.text[:200]}"


@settings(max_examples=30, deadline=10_000)
@given(question=st.text(min_size=1, max_size=300))
def test_query_200_answer_is_non_empty_string(populated_client, question):
    resp = populated_client.post(
        "/query/json",
        json={"question": question},
    )

    if resp.status_code == 200:
        body = resp.json()

        assert "answer" in body
        assert isinstance(body["answer"], str)
        assert len(body["answer"]) > 0


@settings(max_examples=30, deadline=10_000)
@given(question=st.text(min_size=1, max_size=300))
def test_query_200_sources_is_list(populated_client, question):
    resp = populated_client.post(
        "/query/json",
        json={"question": question},
    )

    if resp.status_code == 200:
        body = resp.json()

        assert isinstance(body.get("sources"), list)


@settings(max_examples=20, deadline=10_000)
@given(
    content=st.text(min_size=10, max_size=2000),
    source_id=st.from_regex(r"[a-z][a-z0-9_]{0,49}", fullmatch=True),
)
def test_ingest_chunks_indexed_non_negative(client, content, source_id):
    resp = client.post(
        "/ingest",
        data={
            "text_content": content,
            "source_id": source_id,
        },
    )

    if resp.status_code == 200:
        body = resp.json()

        assert body["chunks_indexed"] >= 0


@settings(max_examples=5, deadline=5_000)
@given(st.none())
def test_root_always_200(client, _none):
    resp = client.get("/")

    assert resp.status_code == 200


@settings(max_examples=5, deadline=5_000)
@given(st.none())
def test_health_always_200(client, _none):
    resp = client.get("/health")

    assert resp.status_code == 200
