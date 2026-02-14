from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


def utcnow():
    return datetime.now(timezone.utc)


class ColumnDefinition(BaseModel):
    name: str
    type: str
    description: Optional[str] = None
    nullable: Optional[bool] = True
    primary_key: Optional[bool] = False


class TableSchema(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    table_name: str = Field(..., alias="table_name")
    description: Optional[str] = None
    columns: List[ColumnDefinition]


class MappingRule(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    source_table: str = Field(..., alias="SourceTable")
    source_column: str = Field(..., alias="SourceColumn")
    target_table: str = Field(..., alias="TargetTable")
    target_column: str = Field(..., alias="TargetColumn")
    transformation: str = Field(..., alias="TransformationParams")


class MigrationContext(BaseModel):
    source_schema: List[TableSchema]
    target_schema: List[TableSchema]
    mappings: List[MappingRule]


class MappingRisk(BaseModel):
    source_field: str
    target_field: str
    risk_type: Literal[
        "type_mismatch", "length_truncation", "null_safety", "semantic_ambiguity"
    ]
    recommendation: str


class SchemaDiffReport(BaseModel):
    safe_mappings: List[MappingRule]
    risky_mappings: List[MappingRisk]
    missing_target_fields: List[str]
    llm_suggestions: List[str]
    timestamp: datetime = Field(default_factory=utcnow)


class SQLGenerationResult(BaseModel):
    sql_script: str
    generation_timestamp: datetime = Field(default_factory=utcnow)
    retry_count: int = 0
    applied_feedback: Optional[List[str]] = None


class ValidationError(BaseModel):
    severity: Literal["ERROR", "WARNING"]
    category: Literal["syntax", "schema", "runtime", "data_quality"]
    message: str
    line_number: Optional[int] = None


class ValidationResult(BaseModel):
    is_valid: bool
    errors: List[ValidationError] = []
    warnings: List[ValidationError] = []
    test_row_count: int = 0
    validation_timestamp: datetime = Field(default_factory=utcnow)
