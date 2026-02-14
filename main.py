#!/usr/bin/env python
"""Entry point for the migration orchestrator."""

from __future__ import annotations

import os

os.environ["LANGCHAIN_TRACING_V2"] = "false"

from pathlib import Path

from core.graph import build_migration_graph
from core.loader import DataLoader
from core.models import MigrationContext
from core.state import MigrationState


def format_validation_report(result) -> str:
    timestamp = result.validation_timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")
    status = "✅ PASSED" if result.is_valid else "❌ FAILED"

    errors_section = ""
    if result.errors:
        errors_section = "\n## Errors\n\n"
        for i, error in enumerate(result.errors, 1):
            errors_section += f"{i}. **[{error.category.upper()}]** {error.message}\n"
    else:
        errors_section = "\n## Errors\n\nNone - validation passed successfully.\n"

    warnings_section = ""
    if result.warnings:
        warnings_section = "\n## Warnings\n\n"
        for i, warning in enumerate(result.warnings, 1):
            warnings_section += (
                f"{i}. **[{warning.category.upper()}]** {warning.message}\n"
            )
    else:
        warnings_section = "\n## Warnings\n\nNone detected.\n"

    report = f"""# Validation Report

**Generated:** {timestamp}
**Status:** {status}
{errors_section}
{warnings_section}
## Test Results

- Test row count: {result.test_row_count}
- Validation timestamp: {timestamp}
"""
    return report


def main():
    print("[*] Starting migration orchestrator...")

    print("\n[>] Loading input data...")
    loader = DataLoader()

    try:
        source_schema = loader.load_schema("data/source_schema.json")
        target_schema = loader.load_schema("data/target_schema.json")
        mappings = loader.load_mappings("data/field_mapping.csv")
        print(f"   [+] Loaded {len(source_schema)} source table(s)")
        print(f"   [+] Loaded {len(target_schema)} target table(s)")
        print(f"   [+] Loaded {len(mappings)} field mapping(s)")
    except Exception as e:
        print(f"   [!] Error loading data: {e}")
        return

    context = MigrationContext(
        source_schema=source_schema, target_schema=target_schema, mappings=mappings
    )

    print("\n[>] Initializing workflow...")
    initial_state = MigrationState(context=context)
    graph = build_migration_graph()

    print("   -> Running Schema Analyst...")
    print("   -> Running SQL Generator...")
    print("   -> Running Validator...")

    try:
        result = graph.invoke(initial_state)
        final_state = MigrationState(**result) if isinstance(result, dict) else result
    except Exception as e:
        print(f"   [!] Workflow error: {e}")
        import traceback
        traceback.print_exc()
        return

    validation_result = final_state.validation_result
    if not validation_result or not validation_result.is_valid:
        print(
            "\n[!] Validation failed after maximum retries. Check validation report for details."
        )
    else:
        print("   -> Running Explainer...")
        print("\n[+] Workflow completed successfully!")

    print("\n[>] Saving outputs...")
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    if final_state.sql_script:
        sql_path = output_dir / "sample_output.sql"
        sql_path.write_text(final_state.sql_script, encoding="utf-8")
        print(f"   [+] SQL script: {sql_path}")

    if final_state.validation_result:
        validation_path = output_dir / "validation_report.md"
        validation_path.write_text(
            format_validation_report(final_state.validation_result), encoding="utf-8"
        )
        print(f"   [+] Validation report: {validation_path}")

    if final_state.explanation:
        explanation_path = output_dir / "sql_explanation.md"
        explanation_path.write_text(final_state.explanation, encoding="utf-8")
        print(f"   [+] Explanation: {explanation_path}")

    print("\n[*] Done. Check the outputs/ directory.")


if __name__ == "__main__":
    main()
