from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import sqlparse

from core.config import get_settings
from core.models import (
    MigrationContext,
    SchemaDiffReport,
    ValidationError,
    ValidationResult,
)


@dataclass
class ParsedSQL:
    target_table: Optional[str] = None
    target_columns: List[str] = None
    source_columns: List[str] = None

    def __post_init__(self):
        if self.target_columns is None:
            self.target_columns = []
        if self.source_columns is None:
            self.source_columns = []


class ValidationAgent:
    def __init__(self, db_connection_string: Optional[str] = None):
        settings = get_settings()
        self.db_url = db_connection_string or settings.DATABASE_URL

    def validate(
        self,
        sql_script: str,
        report: SchemaDiffReport,
        context: MigrationContext,
    ) -> ValidationResult:
        errors: List[ValidationError] = []
        warnings: List[ValidationError] = []

        if not sql_script or not report:
            errors.append(
                ValidationError(
                    severity="ERROR",
                    category="runtime",
                    message="SQL script or schema report is missing.",
                )
            )
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

        syntax_errors = self._check_syntax(sql_script)
        errors.extend(syntax_errors)

        parsed = self._parse_sql(sql_script)
        schema_errors, schema_warnings = self._check_schema_compatibility(
            parsed, context
        )
        errors.extend(schema_errors)
        warnings.extend(schema_warnings)

        is_valid = len(errors) == 0

        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            test_row_count=0,
        )

    def _check_syntax(self, sql: str) -> List[ValidationError]:
        errors: List[ValidationError] = []

        try:
            parsed = sqlparse.parse(sql)
            if not parsed:
                errors.append(
                    ValidationError(
                        severity="ERROR",
                        category="syntax",
                        message="Failed to parse SQL script",
                    )
                )
                return errors
        except Exception as e:
            errors.append(
                ValidationError(
                    severity="ERROR",
                    category="syntax",
                    message=f"SQL parsing error: {str(e)}",
                )
            )
            return errors

        sql_upper = sql.upper()
        begin_count = sql_upper.count("BEGIN")
        commit_count = sql_upper.count("COMMIT")
        rollback_count = sql_upper.count("ROLLBACK")

        if begin_count > 0 and (commit_count + rollback_count) != begin_count:
            errors.append(
                ValidationError(
                    severity="ERROR",
                    category="syntax",
                    message=(
                        f"Unbalanced transactions: {begin_count} BEGIN vs "
                        f"{commit_count} COMMIT + {rollback_count} ROLLBACK"
                    ),
                )
            )

        if "DROP TABLE" in sql_upper or "DROP DATABASE" in sql_upper:
            errors.append(
                ValidationError(
                    severity="ERROR",
                    category="syntax",
                    message="Dangerous operation detected: DROP statement found",
                )
            )

        return errors

    def _parse_sql(self, sql: str) -> ParsedSQL:
        parsed = ParsedSQL()

        insert_pattern = r"INSERT\s+INTO\s+[\"']?(\w+)[\"']?\s*\((.*?)\)"
        insert_match = re.search(insert_pattern, sql, re.IGNORECASE | re.DOTALL)

        if insert_match:
            parsed.target_table = insert_match.group(1)
            columns_str = insert_match.group(2)
            parsed.target_columns = [
                col.strip().strip('"').strip("'")
                for col in columns_str.split(",")
                if col.strip()
            ]

        select_pattern = r"SELECT\s+(.*?)\s+FROM"
        select_match = re.search(select_pattern, sql, re.IGNORECASE | re.DOTALL)

        if select_match:
            select_clause = select_match.group(1)
            source_parts = select_clause.split(",")
            for part in source_parts:
                cleaned = part.strip()
                if " AS " in cleaned.upper():
                    col = cleaned.split("AS")[0].strip()
                else:
                    col = cleaned
                col = col.strip('"').strip("'")
                if col and not col.startswith("("):
                    parsed.source_columns.append(col)

        return parsed

    def _check_schema_compatibility(
        self, parsed: ParsedSQL, context: MigrationContext
    ) -> Tuple[List[ValidationError], List[ValidationError]]:
        errors: List[ValidationError] = []
        warnings: List[ValidationError] = []

        target_schema_cols = {
            col.name for table in context.target_schema for col in table.columns
        }

        source_schema_cols = {
            col.name for table in context.source_schema for col in table.columns
        }

        for col in parsed.target_columns:
            if col not in target_schema_cols:
                errors.append(
                    ValidationError(
                        severity="ERROR",
                        category="schema",
                        message=f"Column '{col}' does not exist in target schema",
                    )
                )

        for col in parsed.source_columns:
            if "(" in col or col.upper() in ("NULL", "CASE", "CAST"):
                continue

            if col not in source_schema_cols:
                warnings.append(
                    ValidationError(
                        severity="WARNING",
                        category="schema",
                        message=f"Column '{col}' not found in source schema (may be transformed)",
                    )
                )

        for risk in context.mappings:
            if not risk.transformation:
                source_col_obj = next(
                    (
                        col
                        for table in context.source_schema
                        for col in table.columns
                        if col.name == risk.source_column
                    ),
                    None,
                )
                target_col_obj = next(
                    (
                        col
                        for table in context.target_schema
                        for col in table.columns
                        if col.name == risk.target_column
                    ),
                    None,
                )

                if source_col_obj and target_col_obj:
                    source_length = self._extract_type_length(source_col_obj.type)
                    target_length = self._extract_type_length(target_col_obj.type)

                    if (
                        source_length
                        and target_length
                        and source_length > target_length
                    ):
                        warnings.append(
                            ValidationError(
                                severity="WARNING",
                                category="data_quality",
                                message=(
                                    f"Potential data truncation: {risk.source_column} "
                                    f"({source_length}) → {risk.target_column} "
                                    f"({target_length})"
                                ),
                            )
                        )

        return errors, warnings

    def _extract_type_length(self, type_str: str) -> Optional[int]:
        match = re.search(r"\((\d+)\)", type_str)
        return int(match.group(1)) if match else None
