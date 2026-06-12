# ── Base ──────────────────────────────────────────────────────────────────────
# python:3.11-slim: ~130MB  |  python:3.11 (sin slim): ~1GB
# "slim" elimina compiladores, docs y herramientas que no usamos en runtime.
FROM python:3.11-slim

# ── uv ────────────────────────────────────────────────────────────────────────
# Copiamos el binario de uv desde su imagen oficial.
# Sin curl, sin scripts de instalación.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# ── Dependencias PRIMERO (layer cache) ───────────────────────────────────────
# El lockfile y pyproject.toml cambian solo cuando actualizamos dependencias.
# El código cambia en cada iteración.
# Poniendo deps antes que código, Docker reutiliza esta capa (CACHED) en
# todos los builds donde solo cambió el código.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev
# --frozen : versiones exactas del lockfile, sin resolver de nuevo
# --no-dev : no instala herramientas de desarrollo (pytest, linters, etc.)

# ── Código ────────────────────────────────────────────────────────────────────
# Va DESPUÉS de las deps para que cambios en el código no invaliden
# la capa de instalación de dependencias.
COPY app/ ./app/
COPY data/ ./data/

# ── Secretos: NUNCA en la imagen ─────────────────────────────────────────────
# NO hacer esto:
#   ENV GOOGLE_API_KEY=AIzaSy...
# docker history <imagen> muestra el contenido de cada capa en texto plano.
# La key se inyecta en runtime: docker run -e GOOGLE_API_KEY=$GOOGLE_API_KEY

# ── Puerto ────────────────────────────────────────────────────────────────────
# EXPOSE documenta el puerto. NO lo hace accesible desde fuera del contenedor.
# La publicación real se hace con -p 8000:8000 en docker run.
EXPOSE 8000

# ── Arranque ──────────────────────────────────────────────────────────────────
# "uv run" asegura que el comando corre dentro del venv gestionado por uv.
# en particular es uv run porque nuestra app es FastAPI, si fuera Flask sería "uv run flask run --host=..."
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
