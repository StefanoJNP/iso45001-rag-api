"""
tests/test_evaluation.py — eval simple para ISO 45001 RAG API

Evalúa que preguntas conocidas devuelvan fuentes asociadas al documento ISO.
"""

GOLD_QUESTIONS = [
    {
        "question": "¿Cuál es el objetivo de la ISO 45001?",
        "expected_source": "iso_45001_2018",
    },
    {
        "question": "¿Qué debe demostrar la alta dirección?",
        "expected_source": "iso_45001_2018",
    },
    {
        "question": "¿Qué dice la norma sobre riesgos y oportunidades?",
        "expected_source": "iso_45001_2018",
    },
]

EVAL_PASS_RATE_THRESHOLD = 0.5


def test_golden_set_has_required_fields():
    for i, item in enumerate(GOLD_QUESTIONS):
        assert "question" in item, f"Entrada {i} no tiene question"
        assert "expected_source" in item, f"Entrada {i} no tiene expected_source"
        assert isinstance(item["question"], str)
        assert isinstance(item["expected_source"], str)
        assert len(item["question"]) > 0
        assert len(item["expected_source"]) > 0


def test_eval_gate(populated_client):
    hits = 0
    misses = []

    for item in GOLD_QUESTIONS:
        resp = populated_client.post(
            "/query/json",
            json={"question": item["question"]},
        )

        if resp.status_code != 200:
            misses.append(
                {
                    "question": item["question"],
                    "expected": item["expected_source"],
                    "actual_status": resp.status_code,
                }
            )
            continue

        body = resp.json()
        sources = body.get("sources", [])

        found = any(item["expected_source"] in source for source in sources)

        if found:
            hits += 1
        else:
            misses.append(
                {
                    "question": item["question"],
                    "expected": item["expected_source"],
                    "retrieved_sources": sources,
                }
            )

    total = len(GOLD_QUESTIONS)
    pass_rate = hits / total if total > 0 else 0.0

    print("\n" + "=" * 60)
    print("EVAL GATE — ISO 45001")
    print("=" * 60)
    print(f"Total preguntas : {total}")
    print(f"Hits            : {hits}")
    print(f"Misses          : {total - hits}")
    print(f"Pass rate       : {pass_rate:.1%}")
    print(f"Umbral          : {EVAL_PASS_RATE_THRESHOLD:.0%}")

    if misses:
        print("\nPreguntas fallidas:")
        for miss in misses:
            print(f"  - {miss['question']!r}")
            print(f"    esperado: {miss.get('expected')!r}")
            print(f"    fuentes : {miss.get('retrieved_sources', 'N/A')}")

    print("=" * 60)

    assert pass_rate >= EVAL_PASS_RATE_THRESHOLD, f"EVAL GATE FALLÓ: pass rate {pass_rate:.1%} < umbral {EVAL_PASS_RATE_THRESHOLD:.0%}"
