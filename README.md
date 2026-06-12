# Control — API RAG ISO 45001:2018

App de IA/ML en contenedor que responde preguntas sobre un documento Markdown de la norma **ISO 45001:2018** usando **FastAPI**, **Gemini**, **ChromaDB** y **Docker**.

La API recibe una pregunta y devuelve una respuesta basada en el documento:

```text
data/auditoria_sbs/iso_45001_2018.md
```

---

## Requisitos

* Docker Desktop
* API key de Google Gemini

La API key no está dentro del código ni dentro de la imagen. Se pasa al ejecutar el contenedor.

---

## Construir la imagen

Desde la carpeta del proyecto:

```cmd
docker build -t iso45001-rag-api .
```

---

## Ejecutar la imagen

```cmd
docker run --rm -p 8000:8000 -e GOOGLE_API_KEY=TU_API_KEY iso45001-rag-api
```

La API quedará disponible en:

```text
http://localhost:8000
```

Documentación interactiva:

```text
http://localhost:8000/docs
```

---

## Probar la API

Health check:

```cmd
curl http://localhost:8000/health
```

Ejemplo de pregunta:

```cmd
curl -X POST "http://localhost:8000/query/json" ^
  -H "Content-Type: application/json" ^
  -d "{\"question\":\"¿Cuál es el objetivo de la ISO 45001?\"}"
```

Respuesta esperada similar:

```json
{
  "answer": "La ISO 45001:2018 establece requisitos para un sistema de gestión de seguridad y salud en el trabajo, con el fin de proporcionar lugares de trabajo seguros y saludables...",
  "sources": [
    "iso_45001_2018 · ISO 45001:2018"
  ],
  "confidence_note": "Respuesta basada en los fragmentos recuperados de la norma ISO 45001:2018."
}
```

---

## Entrega como archivo .tar

Para guardar la imagen Docker como archivo:

```cmd
docker save -o iso45001-rag-api.tar iso45001-rag-api
```

Para cargar la imagen desde el archivo `.tar`:

```cmd
docker load -i iso45001-rag-api.tar
```

Luego ejecutar:

```cmd
docker run --rm -p 8000:8000 -e GOOGLE_API_KEY=TU_API_KEY iso45001-rag-api
```

---

## Tests y lint

Ejecutar tests:

```cmd
uv run pytest -q
```

Ejecutar lint:

```cmd
uv run ruff check .
```

---

## Endpoints principales

```text
GET  /
GET  /health
POST /query/json
POST /query
POST /ingest
GET  /metrics
```

---

## Cumplimiento de la rúbrica

| Criterio                                 | Estado   |
| ---------------------------------------- | -------- |
| Imagen arranca con un solo `docker run`  | Cumplido |
| La IA responde preguntas sobre ISO 45001 | Cumplido |
| README con build, run y ejemplo          | Cumplido |
| Dockerfile slim + uv                     | Cumplido |
| Tests con pytest                         | Cumplido |
| Lint con ruff                            | Cumplido |
| Endpoint `/metrics`                      | Cumplido |
| `.gitignore` y `.dockerignore`           | Cumplido |

---

## Seguridad

La variable `GOOGLE_API_KEY` se inyecta en runtime:

```cmd
-e GOOGLE_API_KEY=TU_API_KEY
```

No se guarda en la imagen Docker, ni en el código, ni en `.env`.
