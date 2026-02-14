from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from core.models import (
    MigrationContext,
    SchemaDiffReport,
    SQLGenerationResult,
)

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel


class MaxRetriesExceeded(Exception):
    pass


class SQLGeneratorAgent:
    SYSTEM_PROMPT = """You are an expert SQL developer specializing in data migrations.
Generate PostgreSQL-compatible INSERT...SELECT statements based on mapping specifications.

CRITICAL RULES:
1. Always wrap operations in BEGIN/COMMIT transactions
2. Use explicit CAST() for type conversions
3. Add COALESCE() for nullable→non-nullable migrations with sensible defaults
4. Comment each transformation with source/target column names
5. Handle dirty data (strip currency symbols, extra whitespace)
6. Use CASE WHEN for conditional logic, never trust raw data
7. For string date conversions, use TO_DATE() or proper parsing
8. For currency cleaning, use REGEXP_REPLACE to remove non-numeric chars

Output ONLY valid SQL. No explanations outside of SQL comments."""

    def __init__(self, llm_client: BaseChatModel, max_retries: int = 3):
        self.llm = llm_client
        self.max_retries = max_retries

    def generate(
        self,
        report: SchemaDiffReport,
        context: MigrationContext,
        validator_feedback: Optional[List[str]] = None,
        retry_count: int = 0,
    ) -> SQLGenerationResult:
        if retry_count >= self.max_retries:
            raise MaxRetriesExceeded(
                f"SQL generation failed after {self.max_retries} attempts"
            )

        if validator_feedback:
            prompt = self._build_retry_prompt(
                report, context, validator_feedback
            )
        else:
            prompt = self._build_initial_prompt(report, context)

        sql_script = self._query_llm(prompt)

        return SQLGenerationResult(
            sql_script=sql_script,
            generation_timestamp=datetime.now(timezone.utc),
            retry_count=retry_count,
            applied_feedback=validator_feedback,
        )

    def _build_initial_prompt(
        self, report: SchemaDiffReport, context: MigrationContext
    ) -> str:
        target_tables = {}
        for mapping in context.mappings:
            if mapping.target_table not in target_tables:
                target_tables[mapping.target_table] = []
            target_tables[mapping.target_table].append(mapping)

        safe_lines = [
            f"- {m.source_table}.{m.source_column} → {m.target_table}.{m.target_column} (Direct)"
            for m in report.safe_mappings
        ]

        risky_lines = [
            f"- {r.source_field} → {r.target_field} ({r.risk_type}): {r.recommendation}"
            for r in report.risky_mappings
        ]

        transform_lines = [
            f"- {m.source_column} → {m.target_column}: {m.transformation}"
            for m in context.mappings
        ]

        prompt = f"""Generate a migration script for the following:

SOURCE SCHEMA:
{self._format_schema(context.source_schema)}

TARGET SCHEMA:
{self._format_schema(context.target_schema)}

SAFE MAPPINGS (direct copy or simple cast):
{chr(10).join(safe_lines) if safe_lines else "None"}

RISKY MAPPINGS (require complex transformations):
{chr(10).join(risky_lines) if risky_lines else "None"}

TRANSFORMATION RULES:
{chr(10).join(transform_lines)}

TARGET TABLES: {', '.join(target_tables.keys())}

Generate complete SQL migration script with:
1. BEGIN/COMMIT transaction wrapper
2. INSERT INTO...SELECT for each target table
3. Proper transformations for risky fields
4. Comments explaining each column mapping
5. NULL safety with COALESCE for required fields

SQL Dialect: PostgreSQL 15"""

        return prompt

    def _build_retry_prompt(
        self,
        report: SchemaDiffReport,
        context: MigrationContext,
        errors: List[str],
    ) -> str:
        initial = self._build_initial_prompt(report, context)

        error_section = f"""
PREVIOUS SQL HAD VALIDATION ERRORS - FIX THE FOLLOWING:
{chr(10).join(f"ERROR {i+1}: {err}" for i, err in enumerate(errors))}

CRITICAL: Address ALL errors above. Regenerate the complete SQL script."""

        return initial + error_section

    def _format_schema(self, schemas) -> str:
        lines = []
        for table in schemas:
            lines.append(f"Table: {table.table_name}")
            for col in table.columns:
                nullable = "NULL" if col.nullable else "NOT NULL"
                lines.append(f"  - {col.name} {col.type} {nullable}")
        return "\n".join(lines)

    def _query_llm(self, prompt: str) -> str:
        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        try:
            response = self.llm.invoke(messages)
            content = response.content
            sql = content.strip()

            if sql.startswith("```sql"):
                sql = sql[6:]
            if sql.startswith("```"):
                sql = sql[3:]
            if sql.endswith("```"):
                sql = sql[:-3]

            return sql.strip()

        except Exception as e:
            return f"""-- ERROR: SQL generation failed
-- {str(e)}

BEGIN;
-- Migration could not be generated
ROLLBACK;
"""
