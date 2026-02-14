from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict

from core.models import (
    MigrationContext,
    SchemaDiffReport,
    ValidationResult,
)


class MigrationState(BaseModel):
    """Shared state passed between LangGraph nodes."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    context: MigrationContext
    schema_report: Optional[SchemaDiffReport] = None
    sql_script: Optional[str] = None
    validation_result: Optional[ValidationResult] = None
    explanation: Optional[str] = None
    validation_errors: List[str] = []
    retry_count: int = 0
