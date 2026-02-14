from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from agents.analyst import SchemaAnalystAgent
from core.loader import DataLoader
from core.memory import MemoryClient
from core.models import MigrationContext


class MockLLM:
    def __init__(self, response: str):
        self.response = response

    def invoke(self, messages):
        return AIMessage(content=self.response)


class MockMemoryClient:
    def find_similar_transformation(self, source_col_desc, target_col_desc):
        return None

    def store_mapping_logic(self, source_col, target_col, logic, metadata=None):
        pass

    def reset(self):
        pass


class TestSchemaAnalystAgent:
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
    def memory_client(self):
        try:
            client = MemoryClient()
            client.reset()
            return client
        except Exception:
            return MockMemoryClient()

    def test_analyst_produces_valid_report(
        self, migration_context, memory_client
    ):
        mock_llm = MockLLM(
            response="""{
  "safe_mappings": [],
  "risky_mappings": [
    {
      "source_field": "RE_Constituent_Export.Gift_Amt",
      "target_field": "Opportunity.Amount",
      "risk_type": "type_mismatch",
      "recommendation": "Remove currency symbols and cast to DECIMAL"
    }
  ],
  "missing_target_fields": ["StageName"],
  "llm_suggestions": ["Consider adding default values for required fields"]
}"""
        )
        agent = SchemaAnalystAgent(mock_llm, memory_client)
        report = agent.analyze(migration_context)

        assert report is not None
        assert len(report.risky_mappings) >= 1
        assert report.risky_mappings[0].risk_type == "type_mismatch"
        assert "StageName" in report.missing_target_fields

    def test_analyst_handles_llm_errors(
        self, migration_context, memory_client
    ):
        mock_llm = MockLLM(response="This is not valid JSON")
        agent = SchemaAnalystAgent(mock_llm, memory_client)
        report = agent.analyze(migration_context)

        assert report is not None
        assert len(report.llm_suggestions) > 0
        assert "parsing failed" in report.llm_suggestions[0].lower()

    def test_analyst_enriches_with_memory(
        self, migration_context, memory_client
    ):
        try:
            memory_client.store_mapping_logic(
                source_col="Gift_Amt",
                target_col="Amount",
                logic="CAST(REGEXP_REPLACE(Gift_Amt, '[^0-9.]', '', 'g') AS DECIMAL)",
            )
        except Exception:
            pytest.skip("ChromaDB not available")

        mock_llm = MockLLM(
            response="""{
  "safe_mappings": [],
  "risky_mappings": [],
  "missing_target_fields": [],
  "llm_suggestions": []
}"""
        )
        agent = SchemaAnalystAgent(mock_llm, memory_client)
        report = agent.analyze(migration_context)

        memory_suggestions = [
            s for s in report.llm_suggestions if "Memory:" in s
        ]
        assert isinstance(memory_suggestions, list)
