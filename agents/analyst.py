from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from pydantic import ValidationError

from core.models import MappingRisk, MappingRule, MigrationContext, SchemaDiffReport

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from core.memory import MemoryClient


class SchemaAnalystAgent:
    SYSTEM_PROMPT = """You are a data migration expert analyzing schema compatibility.
Given a source schema, target schema, and mapping rules, identify:
1. Type mismatches that require transformation
2. Data quality risks (e.g., nullable source → non-nullable target)
3. Semantic meaning differences (e.g., "Amount" vs "Gift_Total")
4. Potential data truncation issues (length mismatches)

Output a structured JSON report following this schema:
{
  "safe_mappings": [
    {"source_table": "...", "source_column": "...", "target_table": "...", "target_column": "...", "transformation": "..."}
  ],
  "risky_mappings": [
    {"source_field": "source_table.column", "target_field": "target_table.column", "risk_type": "type_mismatch|length_truncation|null_safety|semantic_ambiguity", "recommendation": "..."}
  ],
  "missing_target_fields": ["field1", "field2"],
  "llm_suggestions": ["suggestion1", "suggestion2"]
}

IMPORTANT: Return ONLY valid JSON. No markdown formatting, no code blocks."""

    def __init__(self, llm_client: BaseChatModel, memory_client: MemoryClient):
        self.llm = llm_client
        self.memory = memory_client
        self.parser = JsonOutputParser()

    def analyze(self, context: MigrationContext) -> SchemaDiffReport:
        prompt = self._build_analysis_prompt(context)
        llm_response = self._query_llm(prompt)
        return self._enrich_with_memory(llm_response, context)

    def _build_analysis_prompt(self, context: MigrationContext) -> str:
        source_tables = []
        for table in context.source_schema:
            cols = [
                f"  - {col.name} ({col.type}): {col.description or 'N/A'}"
                for col in table.columns
            ]
            source_tables.append(
                f"Table: {table.table_name}\n" + "\n".join(cols)
            )

        target_tables = []
        for table in context.target_schema:
            cols = [
                f"  - {col.name} ({col.type}, nullable={col.nullable}, pk={col.primary_key}): {col.description or 'N/A'}"
                for col in table.columns
            ]
            target_tables.append(
                f"Table: {table.table_name}\n" + "\n".join(cols)
            )

        mapping_lines = [
            f"- {m.source_table}.{m.source_column} → {m.target_table}.{m.target_column} (Transformation: {m.transformation})"
            for m in context.mappings
        ]

        prompt = f"""Analyze the following data migration:

SOURCE SCHEMA:
{chr(10).join(source_tables)}

TARGET SCHEMA:
{chr(10).join(target_tables)}

MAPPING RULES:
{chr(10).join(mapping_lines)}

Analyze compatibility and identify risks. Return JSON only."""

        return prompt

    def _query_llm(self, prompt: str) -> dict:
        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        try:
            response = self.llm.invoke(messages)
            content = response.content
            parsed = json.loads(content)
            return parsed

        except json.JSONDecodeError as e:
            return {
                "safe_mappings": [],
                "risky_mappings": [],
                "missing_target_fields": [],
                "llm_suggestions": [
                    f"LLM response parsing failed: {str(e)}"
                ],
            }
        except Exception as e:
            return {
                "safe_mappings": [],
                "risky_mappings": [],
                "missing_target_fields": [],
                "llm_suggestions": [f"Analysis error: {str(e)}"],
            }

    def _enrich_with_memory(
        self, llm_response: dict, context: MigrationContext
    ) -> SchemaDiffReport:
        suggestions = list(llm_response.get("llm_suggestions", []))

        for mapping in context.mappings:
            try:
                similar = self.memory.find_similar_transformation(
                    source_col_desc=f"{mapping.source_table}.{mapping.source_column}",
                    target_col_desc=f"{mapping.target_table}.{mapping.target_column}",
                )

                if similar:
                    suggestions.append(
                        f"Memory: Similar transformation for {mapping.source_column} → {mapping.target_column}: {similar}"
                    )
            except Exception:
                pass

        try:
            safe_mappings = [
                MappingRule(**m) for m in llm_response.get("safe_mappings", [])
            ]
        except (ValidationError, TypeError):
            safe_mappings = []

        try:
            risky_mappings = [
                MappingRisk(**m)
                for m in llm_response.get("risky_mappings", [])
            ]
        except (ValidationError, TypeError):
            risky_mappings = []

        return SchemaDiffReport(
            safe_mappings=safe_mappings,
            risky_mappings=risky_mappings,
            missing_target_fields=llm_response.get(
                "missing_target_fields", []
            ),
            llm_suggestions=suggestions,
            timestamp=datetime.now(timezone.utc),
        )
