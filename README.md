# AI Data Migration Orchestrator

A **multi-agent system** that generates, validates, and explains SQL migration scripts from source/target schemas and field-mapping metadata. Built for scenarios like migrating CRM data (e.g. Raiser's Edge) to Salesforce Nonprofit Cloud.

---

## Setup Instructions

### 1. Clone and enter the repo

```bash
git clone https://github.com/rishav-goswami/datamigrator.git
cd datamigrator
```

### 2. Environment configuration

Copy the example env file and set your keys:

```bash
cp .env.example .env
```

Edit `.env` and set at least:

- **`LLM_PROVIDER`** – `groq`, `openai`, or `ollama`
- **`GROQ_API_KEY`** – if using Groq (recommended, free tier available)
- **`OPENAI_API_KEY`** – if using OpenAI
- **`OLLAMA_BASE_URL`** – if using local Ollama (default `http://localhost:11434`)

Optional for validation and memory:

- **`DATABASE_URL`** – PostgreSQL URL for validation (e.g. `postgresql://user:password@localhost:5432/migration_db`)
- **`REDIS_URL`** – for LangGraph checkpoints (e.g. `redis://localhost:6379/0`)
- **`CHROMA_HOST`** / **`CHROMA_PORT`** – for ChromaDB (e.g. `localhost:8000`)

### 3. Install dependencies (local run)

```bash
poetry install
```

### 4. Run with Docker (recommended)

Start supporting services and the app:

```bash
docker-compose up db redis vector-db -d   # optional: DB, Redis, ChromaDB
docker-compose up streamlit -d           # Streamlit UI at http://localhost:8501
# or
docker-compose up migrator-app           # CLI one-shot run
```

### 5. Run locally (without Docker)

```bash
poetry install
cp .env.example .env
# Edit .env with your LLM keys
poetry run python main.py                # CLI
# or
poetry run streamlit run streamlit_app.py --server.port=8501   # UI
```

### 6. Input data

Place your migration inputs under `data/`:

| File | Description |
|------|-------------|
| `data/source_schema.json` | Source database schema (tables and columns) |
| `data/target_schema.json` | Target database schema |
| `data/field_mapping.csv` | Field mappings: SourceTable, SourceColumn, TargetTable, TargetColumn, TransformationParams |

Sample files are already in `data/` for a Raiser's Edge → Salesforce-style migration.

---

## Model Used

- **Default:** **Llama 3.3 70B Versatile** (`llama-3.3-70b-versatile`) via **Groq** (fast inference, free tier available).
- **Alternatives:**
  - **OpenAI:** set `LLM_PROVIDER=openai` and `OPENAI_API_KEY`; model is set via `LLM_MODEL` (e.g. `gpt-4o`).
  - **Ollama (local):** set `LLM_PROVIDER=ollama` and run Ollama locally; set `LLM_MODEL` to the model name (e.g. `llama3`, `mistral`).

All four agents (Schema Analyst, SQL Generator, Validation, Explainer) use the same LLM backend configured in `.env`.

---

## How Multi-Agent Coordination Works

Orchestration is implemented with **LangGraph**: a state graph defines the workflow and passes a shared **MigrationState** between agents.

### The four agents

| Agent | Role |
|-------|------|
| **Schema Analyst** | Loads source/target schemas and mapping rules; uses the LLM to classify mappings as safe vs risky, detect type mismatches, and suggest transformations. Can optionally use ChromaDB to recall similar past mappings. |
| **SQL Generator** | Takes the analyst’s report and context and produces a PostgreSQL migration script (e.g. `INSERT ... SELECT`) with proper casts, COALESCE, and comments. Supports **auto-reprompting**: if validation fails, validator errors are fed back and the generator retries (up to 3 times). |
| **Validation Agent** | Checks the generated SQL: syntax (e.g. balanced BEGIN/COMMIT), schema (columns exist in target/source), and basic safety (e.g. no DROP). Can use a real PostgreSQL instance when `DATABASE_URL` is set. |
| **Explainer Agent** | Writes a short, human-readable explanation of the migration (what is moved, which fields are transformed, and why) for audit and handoff. |

### Flow (LangGraph)

1. **START** → **Schema Analyst** (analyzes schemas and mappings).
2. **Schema Analyst** → **SQL Generator** (first draft).
3. **SQL Generator** → **Validation Agent** (syntax + schema checks).
4. **Validator**:
   - If **valid** → **Explainer Agent** → **END** (success).
   - If **invalid** and retries left → back to **SQL Generator** with error feedback.
   - If **invalid** and no retries left → **END** (failure).

State (context, schema report, SQL, validation result, explanation, retry count) is carried through the graph; the validator’s error messages are the “feedback” used for auto-reprompting.

### Optional: memory (ChromaDB)

When ChromaDB is available, the Schema Analyst can store and reuse successful transformation patterns so similar mappings can be suggested in future runs.

---

## Outputs

After a successful run (CLI or Streamlit), outputs are written under `outputs/`:

| File | Description |
|------|-------------|
| `sample_output.sql` | Final generated migration script (PostgreSQL). |
| `validation_report.md` | Validation result: pass/fail, errors, warnings (e.g. type mismatches, missing columns). |
| `sql_explanation.md` | Plain-English explanation of the SQL for auditors and stakeholders. |

---

## Project Layout

```
datamigrator/
├── main.py                 # CLI entry: loads data, runs graph, writes outputs
├── streamlit_app.py       # Streamlit UI: upload schemas/mappings, run migration, view/download results
├── agents/
│   ├── analyst.py         # Schema Analyst Agent
│   ├── sql_generator.py   # SQL Generator Agent
│   ├── validator.py       # Validation Agent
│   └── explainer.py       # Explainer Agent
├── core/
│   ├── config.py         # Settings (env)
│   ├── models.py         # Pydantic models (schemas, reports, state)
│   ├── state.py          # LangGraph MigrationState
│   ├── loader.py         # Load JSON schemas and CSV mappings
│   ├── llm.py            # LLM client (Groq/OpenAI/Ollama)
│   ├── memory.py         # ChromaDB client for semantic memory
│   └── graph.py          # LangGraph workflow definition
├── data/                  # Input schemas and mapping (see above)
├── outputs/               # Generated SQL, validation report, explanation
├── tests/                 # Pytest suite for core, agents, integration
├── devrules/
│   └── ai-case-study.md  # Case study brief
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── .env.example
```

---

## Tests

```bash
poetry install
poetry run pytest tests/ -v
```

---

## License

See repository license (if any).
