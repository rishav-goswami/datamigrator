from __future__ import annotations

import pytest

from core.graph import build_migration_graph
from core.models import ColumnDefinition, MappingRule, MigrationContext, TableSchema
from core.state import MigrationState


class TestIntegration:
    @pytest.fixture
    def sample_migration_context(self) -> MigrationContext:
        source_schema = [
            TableSchema(
                table_name="RE_Constituent_Export",
                columns=[
                    ColumnDefinition(name="Constituent_Name", type="VARCHAR(200)"),
                    ColumnDefinition(name="Email_Address", type="VARCHAR(100)"),
                    ColumnDefinition(name="Gift_Total", type="VARCHAR(50)"),
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
                    ColumnDefinition(
                        name="LastName", type="VARCHAR(100)", nullable=False
                    ),
                    ColumnDefinition(name="Email", type="VARCHAR(255)", nullable=False),
                    ColumnDefinition(
                        name="Total_Lifetime_Giving__c", type="DECIMAL(10,2)"
                    ),
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
            MappingRule(
                source_table="RE_Constituent_Export",
                source_column="Gift_Total",
                target_table="Contact",
                target_column="Total_Lifetime_Giving__c",
                transformation="CAST(REGEXP_REPLACE(Gift_Total, '[^0-9.]', '', 'g') AS DECIMAL)",
            ),
        ]

        return MigrationContext(
            source_schema=source_schema, target_schema=target_schema, mappings=mappings
        )

    def test_graph_construction(self):
        graph = build_migration_graph()
        assert graph is not None

    def test_full_workflow_execution(self, sample_migration_context):
        initial_state = MigrationState(context=sample_migration_context)
        graph = build_migration_graph()
        final_state = graph.invoke(initial_state)

        assert final_state.schema_report is not None
        assert final_state.sql_script is not None
        assert final_state.validation_result is not None

        if final_state.validation_result.is_valid:
            assert final_state.explanation is not None

    def test_workflow_produces_valid_sql(self, sample_migration_context):
        initial_state = MigrationState(context=sample_migration_context)
        graph = build_migration_graph()
        final_state = graph.invoke(initial_state)

        assert "INSERT INTO" in final_state.sql_script.upper()
        assert "SELECT" in final_state.sql_script.upper()

    def test_workflow_validates_sql(self, sample_migration_context):
        initial_state = MigrationState(context=sample_migration_context)
        graph = build_migration_graph()
        final_state = graph.invoke(initial_state)

        assert final_state.validation_result is not None
        assert hasattr(final_state.validation_result, "is_valid")

    def test_workflow_generates_explanation(self, sample_migration_context):
        initial_state = MigrationState(context=sample_migration_context)
        graph = build_migration_graph()
        final_state = graph.invoke(initial_state)

        if final_state.validation_result.is_valid:
            assert final_state.explanation is not None
            assert len(final_state.explanation) > 100
            assert (
                "Constituent" in final_state.explanation
                or "constituent" in final_state.explanation.lower()
            )

    def test_retry_count_increments(self, sample_migration_context):
        initial_state = MigrationState(context=sample_migration_context)
        graph = build_migration_graph()
        final_state = graph.invoke(initial_state)

        assert final_state.retry_count >= 1

    def test_state_persistence(self, sample_migration_context):
        initial_state = MigrationState(context=sample_migration_context)
        graph = build_migration_graph()
        final_state = graph.invoke(initial_state)

        assert final_state.context == initial_state.context
        assert len(final_state.context.mappings) == 3
