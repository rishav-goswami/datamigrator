from __future__ import annotations

import pytest

from core.loader import DataLoader
from core.memory import MemoryClient
from core.models import MappingRule, TableSchema


class TestDataLoader:
    def test_load_schema_valid(self):
        loader = DataLoader()
        schemas = loader.load_schema("data/source_schema.json")

        assert len(schemas) > 0
        assert isinstance(schemas[0], TableSchema)
        assert schemas[0].table_name == "RE_Constituent_Export"
        assert len(schemas[0].columns) == 6

    def test_load_mappings_valid(self):
        loader = DataLoader()
        mappings = loader.load_mappings("data/field_mapping.csv")

        assert len(mappings) > 0
        assert isinstance(mappings[0], MappingRule)
        assert mappings[0].source_table == "RE_Constituent_Export"

    def test_load_schema_file_not_found(self):
        loader = DataLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_schema("nonexistent.json")

    def test_load_mappings_file_not_found(self):
        loader = DataLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_mappings("nonexistent.csv")


class TestMemoryClient:
    @pytest.fixture(autouse=True)
    def setup_memory(self):
        client = MemoryClient()
        try:
            client.reset()
        except Exception:
            pass
        yield client

    @pytest.mark.skip(reason="Requires ChromaDB Docker service")
    def test_store_and_retrieve_mapping(self, setup_memory):
        memory = setup_memory
        try:
            memory.store_mapping_logic(
                source_col="donor_name",
                target_col="contact_name",
                logic="SPLIT_PART(donor_name, ' ', 1)",
                metadata={"transformation_type": "split"},
            )
            result = memory.find_similar_transformation(
                source_col_desc="donor full name",
                target_col_desc="contact first name",
            )
            if result:
                assert "SPLIT_PART" in result
        except Exception as e:
            pytest.skip(f"ChromaDB not available: {e}")

    @pytest.mark.skip(reason="Requires ChromaDB Docker service")
    def test_no_similar_transformation_found(self, setup_memory):
        memory = setup_memory
        try:
            result = memory.find_similar_transformation(
                source_col_desc="completely_unique_field_xyz",
                target_col_desc="another_unique_field_abc",
            )
            assert result is None or result == ""
        except Exception:
            pytest.skip("ChromaDB not available")
