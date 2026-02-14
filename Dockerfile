FROM python:3.13-slim

WORKDIR /app

COPY pyproject.toml ./

RUN --mount=type=cache,target=/root/.cache/pip \
    python -c "import tomllib; \
    deps = tomllib.load(open('pyproject.toml', 'rb'))['project']['dependencies']; \
    import subprocess; \
    subprocess.run(['pip', 'install', '--no-cache-dir'] + deps, check=True)"

RUN python - <<'PY'
import sys
try:
    from chromadb.utils import embedding_functions as ef
    embed_fn = ef.DefaultEmbeddingFunction()
    embed_fn(['warmup', 'test', 'embedding'])
except Exception as exc:
    print(f'Chroma warmup failed: {exc}', file=sys.stderr)
PY

COPY . .

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

CMD ["python", "main.py"]
