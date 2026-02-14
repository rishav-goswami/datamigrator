from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List

from pydantic import ValidationError

from .models import MappingRule, TableSchema


class DataLoader:
    @staticmethod
    def load_schema(file_path: str) -> List[TableSchema]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Schema file not found: {file_path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        try:
            tables = [TableSchema(**table) for table in data.get("tables", [])]
            return tables
        except ValidationError as e:
            raise ValidationError(
                f"Invalid schema format in {file_path}: {e}"
            ) from e

    @staticmethod
    def load_mappings(file_path: str) -> List[MappingRule]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Mapping file not found: {file_path}")

        mappings = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    mapping = MappingRule(**row)
                    mappings.append(mapping)
                except ValidationError as e:
                    raise ValidationError(
                        f"Invalid mapping row in {file_path}: {row}. Error: {e}"
                    ) from e

        return mappings
