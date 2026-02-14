# Data Migration Orchestrator

Multi-agent system that generates, validates, and explains SQL migration scripts from schema and mapping metadata.

## Quick Start (Streamlit)

1. Copy `.env.example` to `.env` and set `LLM_PROVIDER`, `GROQ_API_KEY` (or OpenAI/Ollama).
2. Run: `docker-compose up streamlit -d`
3. Open http://localhost:8501, upload schemas and mappings, then click Run Migration.

## Quick Start (CLI)

1. Configure `.env` as above.
2. Run: `docker-compose up migrator-app`
3. Outputs appear in `outputs/`: `sample_output.sql`, `validation_report.md`, `sql_explanation.md`.

## Local Development

```bash
poetry install
cp .env.example .env
docker-compose up db redis vector-db -d
poetry run python main.py
poetry run pytest
```

## Input Data

Put in `data/`:
- `source_schema.json` – source database schema
- `target_schema.json` – target schema
- `field_mapping.csv` – field-level mappings (SourceTable, SourceColumn, TargetTable, TargetColumn, TransformationParams)

## Model

Default: `llama-3.3-70b-versatile` via Groq. Alternatives: set `LLM_PROVIDER=openai` or `ollama`.

## Coordination

- LangGraph state graph: Schema Analyst → SQL Generator → Validator → (retry or) Explainer.
- Validator errors are fed back to the generator (max 3 retries).
- Successful mappings can be stored in ChromaDB for reuse.

## Outputs

- `sample_output.sql` – generated migration script
- `validation_report.md` – validation result
- `sql_explanation.md` – plain-language explanation
