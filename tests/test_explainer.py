from __future__ import annotations

from unittest.mock import Mock

import pytest

from agents.explainer import ExplainerAgent
from core.models import (
    ColumnDefinition,
    MappingRule,
    MappingRisk,
    MigrationContext,
    SchemaDiffReport,
    TableSchema,
    ValidationError,
    ValidationResult,
)


class TestExplainerAgent:
    @pytest.fixture
    def mock_llm(self):
        mock = Mock()
        mock_response = Mock()
        mock_response.content = """
## Overview
This migration transfers donor data from a legacy CRM to Salesforce.

## Field Transformations
- **Constituent_Name** → **FirstName**: Extract first word from full name
- **Email_Address** → **Email**: Direct copy with validation

## Data Quality Notes
All validations passed successfully.
"""
        mock.invoke.return_value = mock_response
        return mock

    @pytest.fixture
    def sample_context(self) -> MigrationContext:
        source_schema = [
            TableSchema(
                table_name="RE_Constituent_Export",
                columns=[
                    ColumnDefinition(name="Constituent_Name", type="VARCHAR(200)"),
                    ColumnDefinition(name="Email_Address", type="VARCHAR(100)"),
                ],
            )
        ]

        target_schema = [
            TableSchema(
                table_name="Contact",
                columns=[
                    ColumnDefinition(
                        name="FirstName", type="VARCHAR(100)", nullable=False
                    ),
                    ColumnDefinition(name="Email", type="VARCHAR(255)", nullable=False),
                ],
            )
        ]

        mappings = [
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
        ]

        return MigrationContext(
            source_schema=source_schema, target_schema=target_schema, mappings=mappings
        )

    @pytest.fixture
    def sample_report(self) -> SchemaDiffReport:
        return SchemaDiffReport(
            safe_mappings=[],
            risky_mappings=[
                MappingRisk(
                    source_field="Constituent_Name",
                    target_field="FirstName",
                    risk_type="length_truncation",
                    recommendation="Use SUBSTRING to enforce length limit",
                )
            ],
            missing_target_fields=[],
            llm_suggestions=["Split full names appropriately"],
        )

    @pytest.fixture
    def sample_validation(self) -> ValidationResult:
        return ValidationResult(
            is_valid=True,
            errors=[],
            warnings=[
                ValidationError(
                    severity="WARNING",
                    category="data_quality",
                    message="Potential data truncation for long names",
                )
            ],
        )

    def test_explainer_generates_markdown(
        self, mock_llm, sample_context, sample_report, sample_validation
    ):
        sql = "INSERT INTO Contact SELECT * FROM RE_Constituent_Export;"
        agent = ExplainerAgent(mock_llm)
        result = agent.explain(sql, sample_report, sample_validation, sample_context)

        assert result is not None
        assert isinstance(result, str)
        assert "# Migration Explanation" in result
        assert "Generated:" in result

    def test_explainer_includes_header(
        self, mock_llm, sample_context, sample_report, sample_validation
    ):
        sql = "INSERT INTO Contact SELECT * FROM RE_Constituent_Export;"
        agent = ExplainerAgent(mock_llm)
        result = agent.explain(sql, sample_report, sample_validation, sample_context)

        assert "Generated:" in result
        assert "UTC" in result

    def test_explainer_calls_llm(
        self, mock_llm, sample_context, sample_report, sample_validation
    ):
        sql = "INSERT INTO Contact SELECT * FROM RE_Constituent_Export;"
        agent = ExplainerAgent(mock_llm)
        agent.explain(sql, sample_report, sample_validation, sample_context)

        assert mock_llm.invoke.called

    def test_explainer_handles_llm_errors(
        self, sample_context, sample_report, sample_validation
    ):
        mock_llm = Mock()
        mock_llm.invoke.side_effect = Exception("LLM connection failed")

        sql = "INSERT INTO Contact SELECT * FROM RE_Constituent_Export;"
        agent = ExplainerAgent(mock_llm)
        result = agent.explain(sql, sample_report, sample_validation, sample_context)

        assert result is not None
        assert "Migration Explanation" in result
        assert "LLM error" in result or "Automatic explanation failed" in result

    def test_fallback_explanation_includes_sql(
        self, sample_context, sample_report, sample_validation
    ):
        mock_llm = Mock()
        mock_llm.invoke.side_effect = Exception("Test error")

        sql = "INSERT INTO Contact (FirstName) VALUES ('Test');"
        agent = ExplainerAgent(mock_llm)
        result = agent.explain(sql, sample_report, sample_validation, sample_context)

        assert "INSERT INTO Contact" in result
