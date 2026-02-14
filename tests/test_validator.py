from __future__ import annotations

import pytest

from agents.validator import ValidationAgent
from core.models import (
    ColumnDefinition,
    MappingRule,
    MigrationContext,
    SchemaDiffReport,
    TableSchema,
)


class TestValidationAgent:
    @pytest.fixture
    def sample_context(self) -> MigrationContext:
        source_schema = [
            TableSchema(
                table_name="RE_Constituent_Export",
                columns=[
                    ColumnDefinition(
                        name="Constituent_Name",
                        type="VARCHAR(200)",
                        nullable=True,
                    ),
                    ColumnDefinition(
                        name="Email_Address",
                        type="VARCHAR(100)",
                        nullable=True,
                    ),
                    ColumnDefinition(
                        name="Gift_Total",
                        type="VARCHAR(50)",
                        nullable=True,
                    ),
                ],
            )
        ]

        target_schema = [
            TableSchema(
                table_name="Contact",
                columns=[
                    ColumnDefinition(
                        name="FirstName",
                        type="VARCHAR(100)",
                        nullable=False,
                    ),
                    ColumnDefinition(
                        name="LastName",
                        type="VARCHAR(100)",
                        nullable=False,
                    ),
                    ColumnDefinition(
                        name="Email",
                        type="VARCHAR(255)",
                        nullable=False,
                    ),
                    ColumnDefinition(
                        name="Total_Lifetime_Giving__c",
                        type="DECIMAL(10,2)",
                        nullable=True,
                    ),
                ],
            )
        ]

        mapping_rules = [
            MappingRule(
                source_table="RE_Constituent_Export",
                source_column="Constituent_Name",
                target_table="Contact",
                target_column="FirstName",
                transformation="SPLIT_PART(Constituent_Name, ' ', 1)",
            ),
            MappingRule(
                source_table="RE_Constituent_Export",
                source_column="Email_Address",
                target_table="Contact",
                target_column="Email",
                transformation="Email_Address",
            ),
            MappingRule(
                source_table="RE_Constituent_Export",
                source_column="Gift_Total",
                target_table="Contact",
                target_column="Total_Lifetime_Giving__c",
                transformation="CAST(REGEXP_REPLACE(Gift_Total, '[^0-9.]', '', 'g') AS DECIMAL)",
            ),
        ]

        return MigrationContext(
            source_schema=source_schema,
            target_schema=target_schema,
            mappings=mapping_rules,
        )

    @pytest.fixture
    def sample_report(self) -> SchemaDiffReport:
        return SchemaDiffReport(
            safe_mappings=[],
            risky_mappings=[],
            missing_target_fields=["LastName"],
            llm_suggestions=["Use SPLIT_PART for name parsing"],
        )

    def test_valid_sql_passes(self, sample_context, sample_report):
        valid_sql = """
BEGIN;

INSERT INTO "Contact" ("FirstName", "LastName", "Email", "Total_Lifetime_Giving__c")
SELECT 
    SPLIT_PART("Constituent_Name", ' ', 1) AS "FirstName",
    COALESCE(SPLIT_PART("Constituent_Name", ' ', 2), 'Unknown') AS "LastName",
    "Email_Address" AS "Email",
    CAST(REGEXP_REPLACE("Gift_Total", '[^0-9.]', '', 'g') AS DECIMAL) AS "Total_Lifetime_Giving__c"
FROM "RE_Constituent_Export";

COMMIT;
"""
        validator = ValidationAgent()
        result = validator.validate(valid_sql, sample_report, sample_context)

        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_syntax_error_caught(self, sample_context, sample_report):
        bad_sql = """
BEGIN;

INSERT INTO Contact (FirstName, Email
SELECT FirstName, Email FROM Source

COMMIT;
"""
        validator = ValidationAgent()
        result = validator.validate(bad_sql, sample_report, sample_context)

        assert result.is_valid is False or len(result.warnings) > 0

    def test_missing_column_detected(self, sample_context, sample_report):
        sql_with_fake_column = """
BEGIN;

INSERT INTO "Contact" ("FirstName", "FakeColumn", "Email")
SELECT "Constituent_Name", "NonExistent", "Email_Address"
FROM "RE_Constituent_Export";

COMMIT;
"""
        validator = ValidationAgent()
        result = validator.validate(sql_with_fake_column, sample_report, sample_context)

        assert result.is_valid is False
        error_messages = [e.message for e in result.errors]
        assert any("FakeColumn" in msg for msg in error_messages)

    def test_unbalanced_transactions_detected(self, sample_context, sample_report):
        unbalanced_sql = """
BEGIN;

INSERT INTO "Contact" ("FirstName", "Email")
SELECT "Constituent_Name", "Email_Address"
FROM "RE_Constituent_Export";

"""
        validator = ValidationAgent()
        result = validator.validate(unbalanced_sql, sample_report, sample_context)

        assert result.is_valid is False
        error_messages = [e.message for e in result.errors]
        assert any("Unbalanced" in msg or "transaction" in msg.lower() for msg in error_messages)

    def test_dangerous_operations_blocked(self, sample_context, sample_report):
        dangerous_sql = """
DROP TABLE "Contact";

INSERT INTO "Contact" ("FirstName", "Email")
SELECT "Constituent_Name", "Email_Address"
FROM "RE_Constituent_Export";
"""
        validator = ValidationAgent()
        result = validator.validate(dangerous_sql, sample_report, sample_context)

        assert result.is_valid is False
        error_messages = [e.message for e in result.errors]
        assert any("DROP" in msg or "Dangerous" in msg for msg in error_messages)

    def test_data_truncation_warning(self, sample_context, sample_report):
        sample_context.mappings.append(
            MappingRule(
                source_table="RE_Constituent_Export",
                source_column="Constituent_Name",
                target_table="Contact",
                target_column="FirstName",
                transformation="",
            )
        )

        sql = """
BEGIN;

INSERT INTO "Contact" ("FirstName", "Email")
SELECT "Constituent_Name", "Email_Address"
FROM "RE_Constituent_Export";

COMMIT;
"""
        validator = ValidationAgent()
        result = validator.validate(sql, sample_report, sample_context)

        warning_messages = [w.message for w in result.warnings]
        assert any("truncation" in msg.lower() for msg in warning_messages)

    def test_validation_result_structure(self, sample_context, sample_report):
        sql = """
BEGIN;
INSERT INTO "Contact" ("FirstName", "Email")
SELECT "Constituent_Name", "Email_Address"
FROM "RE_Constituent_Export";
COMMIT;
"""
        validator = ValidationAgent()
        result = validator.validate(sql, sample_report, sample_context)

        assert hasattr(result, "is_valid")
        assert hasattr(result, "errors")
        assert hasattr(result, "warnings")
        assert hasattr(result, "test_row_count")
        assert hasattr(result, "validation_timestamp")
        assert isinstance(result.is_valid, bool)
        assert isinstance(result.errors, list)
        assert isinstance(result.warnings, list)
        assert isinstance(result.test_row_count, int)

    def test_error_categorization(self, sample_context, sample_report):
        sql_with_multiple_issues = """
BEGIN;

INSERT INTO "Contact" ("FakeColumn", "Email")
SELECT "Constituent_Name", "Email_Address"
FROM "RE_Constituent_Export";
"""
        validator = ValidationAgent()
        result = validator.validate(sql_with_multiple_issues, sample_report, sample_context)

        categories = {e.category for e in result.errors}
        assert "syntax" in categories or "schema" in categories
