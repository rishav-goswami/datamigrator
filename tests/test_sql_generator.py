from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from agents.sql_generator import MaxRetriesExceeded, SQLGeneratorAgent
from core.loader import DataLoader
from core.models import (
    MappingRisk,
    MigrationContext,
    SchemaDiffReport,
)


class MockLLM:
    def __init__(self, response: str):
        self.response = response
        self.call_count = 0

    def invoke(self, messages):
        self.call_count += 1
        return AIMessage(content=self.response)


class TestSQLGeneratorAgent:
    @pytest.fixture
    def migration_context(self):
        loader = DataLoader()
        source = loader.load_schema("data/source_schema.json")
        target = loader.load_schema("data/target_schema.json")
        mappings = loader.load_mappings("data/field_mapping.csv")
        return MigrationContext(
            source_schema=source, target_schema=target, mappings=mappings
        )

    @pytest.fixture
    def schema_report(self, migration_context):
        return SchemaDiffReport(
            safe_mappings=[],
            risky_mappings=[
                MappingRisk(
                    source_field="RE_Constituent_Export.Gift_Amt",
                    target_field="Opportunity.Amount",
                    risk_type="type_mismatch",
                    recommendation="Remove $ and cast to DECIMAL",
                )
            ],
            missing_target_fields=["StageName"],
            llm_suggestions=[],
        )

    def test_basic_sql_generation(self, migration_context, schema_report):
        mock_llm = MockLLM(
            response="""BEGIN;

INSERT INTO "Contact" ("Legacy_Id__c", "FirstName", "LastName", "CreatedDate")
SELECT 
    "Cons_ID",
    SPLIT_PART("Full_Name", ' ', 1),
    SPLIT_PART("Full_Name", ' ', 2),
    TO_DATE("Entry_Date", 'MM/DD/YYYY')
FROM "RE_Constituent_Export";

COMMIT;"""
        )
        agent = SQLGeneratorAgent(mock_llm)
        result = agent.generate(schema_report, migration_context)

        assert result is not None
        assert "INSERT INTO" in result.sql_script
        assert "BEGIN" in result.sql_script
        assert "COMMIT" in result.sql_script
        assert result.retry_count == 0
        assert result.applied_feedback is None

    def test_complex_transformation(self, migration_context, schema_report):
        mock_llm = MockLLM(
            response="""BEGIN;

INSERT INTO "Opportunity" ("Name", "Amount", "CloseDate")
SELECT 
    CONCAT('Donation - ', "Campaign_Code"),
    CAST(REGEXP_REPLACE("Gift_Amt", '[^0-9.]', '', 'g') AS DECIMAL(18,2)),
    TO_DATE("Gift_Date", 'MM/DD/YYYY')
FROM "RE_Constituent_Export";

COMMIT;"""
        )
        agent = SQLGeneratorAgent(mock_llm)
        result = agent.generate(schema_report, migration_context)

        assert "REGEXP_REPLACE" in result.sql_script
        assert "CAST" in result.sql_script
        assert "DECIMAL" in result.sql_script

    def test_retry_with_feedback(self, migration_context, schema_report):
        mock_llm = MockLLM(
            response="""BEGIN;

INSERT INTO "Contact" ("Legacy_Id__c", "FirstName", "LastName", "CreatedDate")
SELECT 
    "Cons_ID",
    COALESCE(SPLIT_PART("Full_Name", ' ', 1), 'Unknown'),
    COALESCE(SPLIT_PART("Full_Name", ' ', 2), 'Unknown'),
    TO_DATE("Entry_Date", 'MM/DD/YYYY')
FROM "RE_Constituent_Export";

COMMIT;"""
        )
        agent = SQLGeneratorAgent(mock_llm)
        result1 = agent.generate(schema_report, migration_context)
        feedback = ["Column LastName violates NOT NULL constraint"]
        result2 = agent.generate(
            schema_report, migration_context, validator_feedback=feedback, retry_count=1
        )

        assert result2.retry_count == 1
        assert result2.applied_feedback == feedback
        assert "COALESCE" in result2.sql_script

    def test_max_retry_exceeded(self, migration_context, schema_report):
        mock_llm = MockLLM(response="SELECT 1;")
        agent = SQLGeneratorAgent(mock_llm, max_retries=3)

        with pytest.raises(MaxRetriesExceeded):
            agent.generate(
                schema_report,
                migration_context,
                validator_feedback=["Error"],
                retry_count=3,
            )

    def test_sql_cleanup(self, migration_context, schema_report):
        mock_llm = MockLLM(
            response="""```sql
BEGIN;
SELECT 1;
COMMIT;
```"""
        )
        agent = SQLGeneratorAgent(mock_llm)
        result = agent.generate(schema_report, migration_context)

        assert "```" not in result.sql_script
        assert "BEGIN" in result.sql_script

    def test_llm_error_handling(self, migration_context, schema_report):
        class ErrorLLM:
            def invoke(self, messages):
                raise Exception("LLM API error")

        agent = SQLGeneratorAgent(ErrorLLM())
        result = agent.generate(schema_report, migration_context)

        assert "ERROR" in result.sql_script
        assert "ROLLBACK" in result.sql_script
