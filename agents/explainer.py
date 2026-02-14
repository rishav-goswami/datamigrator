from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from core.models import MigrationContext, SchemaDiffReport, ValidationResult


class ExplainerAgent:
    SYSTEM_PROMPT = """You are a technical writer creating documentation for non-technical stakeholders.
Given a SQL migration script, explain each transformation in simple business terms.

GUIDELINES:
1. Avoid technical jargon - instead of "CAST", say "convert to number"
2. Explain the "why" behind each transformation clearly
3. Highlight data quality risks in layman's terms
4. Use tables and bullet lists for clarity
5. Output in clean Markdown format
6. Be specific about what fields map to what

TONE: Professional but accessible. Imagine explaining to a project manager or business analyst.
"""

    def __init__(self, llm_client: Any):
        self.llm = llm_client

    def explain(
        self,
        sql_script: str,
        report: SchemaDiffReport,
        validation: ValidationResult,
        context: MigrationContext,
    ) -> str:
        prompt = self._build_explanation_prompt(
            sql_script, report, validation, context
        )

        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        try:
            response = self.llm.invoke(messages)
            explanation = response.content

            timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            header = f"# Migration Explanation\n\n**Generated:** {timestamp}\n\n"

            return header + explanation

        except Exception as e:
            return self._generate_fallback_explanation(sql_script, context, str(e))

    def _build_explanation_prompt(
        self,
        sql_script: str,
        report: SchemaDiffReport,
        validation: ValidationResult,
        context: MigrationContext,
    ) -> str:
        mappings_summary = "\n".join(
            [
                f"- {m.source_column} → {m.target_column}: {m.transformation}"
                for m in context.mappings
            ]
        )

        risky_summary = "\n".join(
            [
                f"- {r.source_field} → {r.target_field}: {r.risk_type} - {r.recommendation}"
                for r in report.risky_mappings
            ]
        )
        if not risky_summary:
            risky_summary = "None identified"

        warnings_summary = "\n".join(
            [f"- {w.category.upper()}: {w.message}" for w in validation.warnings]
        )
        if not warnings_summary:
            warnings_summary = "None - all validations passed"

        source_tables = ", ".join([t.table_name for t in context.source_schema])
        target_tables = ", ".join([t.table_name for t in context.target_schema])

        prompt = f"""Explain the following data migration script:

**Source Tables:** {source_tables}
**Target Tables:** {target_tables}

**SQL Script:**
```sql
{sql_script}
```

**Field Mappings:**
{mappings_summary}

**Risky Transformations Identified:**
{risky_summary}

**Validation Warnings:**
{warnings_summary}

Generate a complete migration explanation document in Markdown format that:
1. Provides an overview of what data is being migrated
2. Explains each field transformation in simple terms
3. Documents any data quality concerns
4. Summarizes validation results
5. Provides next steps for execution

Make it comprehensive but easy to understand for non-technical stakeholders.
"""
        return prompt

    def _generate_fallback_explanation(
        self, sql_script: str, context: MigrationContext, error: str
    ) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        source_tables = ", ".join([t.table_name for t in context.source_schema])
        target_tables = ", ".join([t.table_name for t in context.target_schema])

        mappings_list = "\n".join(
            [
                f"- **{m.source_column}** → **{m.target_column}**"
                for m in context.mappings
            ]
        )

        return f"""# Migration Explanation

**Generated:** {timestamp}
**Status:** ⚠️ Automatic explanation failed (LLM error: {error})

## Overview
This migration script transfers data from **{source_tables}** to **{target_tables}**.

## Field Mappings
{mappings_list}

## SQL Script
```sql
{sql_script}
```

## Notes
This is a basic explanation generated automatically. For detailed analysis, please review the SQL script and validation report manually.
"""
