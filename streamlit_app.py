"""
Streamlit UI for AI Data Migration Orchestrator

Interactive web interface for uploading schemas, configuring agents,
monitoring migration progress, and downloading outputs.
"""

import json
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional
import base64

import streamlit as st
import streamlit.components.v1 as components

# Disable LangSmith tracing to avoid slow network calls during import
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")

# Configure page
st.set_page_config(
    page_title="AI Data Migration Orchestrator",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import core components (after Streamlit config)
from core.graph import build_migration_graph
from core.loader import DataLoader
from core.models import MigrationContext
from core.state import MigrationState


def init_session_state():
    """Initialize session state variables"""
    if 'migration_complete' not in st.session_state:
        st.session_state.migration_complete = False
    if 'outputs' not in st.session_state:
        st.session_state.outputs = {}
    if 'error' not in st.session_state:
        st.session_state.error = None
    if 'events' not in st.session_state:
        st.session_state.events = []
    if 'execution_path' not in st.session_state:
        st.session_state.execution_path = []
    if 'embedding_loaded' not in st.session_state:
        st.session_state.embedding_loaded = False


def warmup_embedding_model():
    """Preload ChromaDB embedding model on first app startup."""
    if not st.session_state.embedding_loaded:
        try:
            with st.spinner('🔄 Loading embedding model (first run only)...'):
                from chromadb.utils import embedding_functions as ef
                # Trigger model download/load
                embed_fn = ef.DefaultEmbeddingFunction()
                embed_fn(['warmup'])
                st.session_state.embedding_loaded = True
        except Exception as e:
            # Don't block the app if warmup fails
            st.warning(f"Embedding model warmup skipped: {e}")


def validate_json_file(uploaded_file) -> Optional[Dict]:
    """Validate and parse uploaded JSON file"""
    try:
        content = uploaded_file.read()
        uploaded_file.seek(0)
        return json.loads(content)
    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON in {uploaded_file.name}: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Error reading {uploaded_file.name}: {str(e)}")
        return None


def validate_csv_file(uploaded_file) -> bool:
    """Validate uploaded CSV file has required columns"""
    try:
        import pandas as pd
        uploaded_file.seek(0)
        
        df = pd.read_csv(uploaded_file)
        required_cols = [
            "SourceTable",
            "SourceColumn",
            "TargetTable",
            "TargetColumn",
            "TransformationParams",
        ]
        
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            st.error(f"CSV missing required columns: {', '.join(missing)}")
            return False
        return True
    except Exception as e:
        st.error(f"Error validating CSV: {str(e)}")
        return False


def save_uploaded_file(uploaded_file, destination: Path) -> bool:
    """Save uploaded file to destination path"""
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with open(destination, 'wb') as f:
            f.write(uploaded_file.getvalue())
        return True
    except Exception as e:
        st.error(f"Error saving {uploaded_file.name}: {str(e)}")
        return False


def format_validation_report(result) -> str:
    """Convert ValidationResult to Markdown report."""

    if result is None:
        return "# Validation Report\n\nNo validation result available."

    timestamp = result.validation_timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")
    status = "✅ PASSED" if result.is_valid else "❌ FAILED"

    errors_section = "\n## Errors\n\n"
    if result.errors:
        for i, error in enumerate(result.errors, 1):
            errors_section += f"{i}. **[{error.category.upper()}]** {error.message}\n"
    else:
        errors_section += "None - validation passed successfully.\n"

    warnings_section = "\n## Warnings\n\n"
    if result.warnings:
        for i, warning in enumerate(result.warnings, 1):
            warnings_section += (
                f"{i}. **[{warning.category.upper()}]** {warning.message}\n"
            )
    else:
        warnings_section += "None detected.\n"

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


def build_agent_activity(final_state: MigrationState) -> Dict[str, str]:
    """Create high-level summaries of each agent's work."""

    activity = {}

    schema_report = final_state.schema_report
    if schema_report:
        activity["Schema Analyst"] = (
            f"Safe mappings: {len(schema_report.safe_mappings)}; "
            f"Risky mappings: {len(schema_report.risky_mappings)}; "
            f"Missing target fields: {len(schema_report.missing_target_fields)}; "
            f"Suggestions: {len(schema_report.llm_suggestions)}"
        )
    else:
        activity["Schema Analyst"] = "No schema report available."

    sql_script = final_state.sql_script or ""
    sql_lines = len(sql_script.splitlines()) if sql_script else 0
    activity["SQL Generator"] = (
        f"Generated SQL length: {len(sql_script)} chars; Lines: {sql_lines}"
    )

    validation_result = final_state.validation_result
    if validation_result:
        activity["Validation Agent"] = (
            f"Valid: {validation_result.is_valid}; "
            f"Errors: {len(validation_result.errors)}; "
            f"Warnings: {len(validation_result.warnings)}"
        )
    else:
        activity["Validation Agent"] = "No validation result available."

    explanation = final_state.explanation or ""
    activity["Explainer Agent"] = (
        f"Explanation length: {len(explanation)} chars"
    )

    return activity


def preview_text(value: str, max_lines: int = 40) -> str:
    """Limit long text previews for UI display."""

    if not value:
        return ""
    lines = value.splitlines()
    if len(lines) <= max_lines:
        return value
    return "\n".join(lines[:max_lines]) + "\n... (truncated)"


def generate_graph_mermaid() -> str:
    """Generate Mermaid diagram for the orchestration graph."""
    return """
graph TD
    START([START]) --> analyst[Schema Analyst]
    analyst --> generator[SQL Generator]
    generator --> validator[Validator]
    validator -->|Valid| explainer[Explainer]
    validator -->|Invalid & retries < 3| generator
    validator -->|Invalid & retries >= 3| FAIL([END - FAILED])
    explainer --> END([END - SUCCESS])
    
    style analyst fill:#e1f5ff,stroke:#01579b
    style generator fill:#fff9c4,stroke:#f57f17
    style validator fill:#f3e5f5,stroke:#4a148c
    style explainer fill:#e8f5e9,stroke:#1b5e20
    style START fill:#90caf9,stroke:#0d47a1
    style END fill:#a5d6a7,stroke:#2e7d32
    style FAIL fill:#ef9a9a,stroke:#c62828
"""


def generate_execution_path_mermaid(execution_path: list) -> str:
    """Generate Mermaid diagram showing the actual execution path taken."""
    if not execution_path:
        return """
graph TD
    START([No execution yet])
    style START fill:#bdbdbd,stroke:#424242
"""
    
    mermaid = "graph TD\\n"
    mermaid += "    START([START]) --> node0[" + execution_path[0] + "]\\n"
    
    for i in range(len(execution_path) - 1):
        current = execution_path[i]
        next_node = execution_path[i + 1]
        mermaid += f"    node{i}[{current}] --> node{i+1}[{next_node}]\\n"
    
    # Add END node
    last_idx = len(execution_path) - 1
    mermaid += f"    node{last_idx}[{execution_path[last_idx]}] --> END([END])\\n\\n"
    
    # Styling
    for i, node in enumerate(execution_path):
        if "analyst" in node.lower():
            mermaid += f"    style node{i} fill:#e1f5ff,stroke:#01579b\\n"
        elif "generator" in node.lower():
            mermaid += f"    style node{i} fill:#fff9c4,stroke:#f57f17\\n"
        elif "validator" in node.lower():
            if "passed" in node.lower():
                mermaid += f"    style node{i} fill:#e8f5e9,stroke:#1b5e20\\n"
            else:
                mermaid += f"    style node{i} fill:#ffccbc,stroke:#bf360c\\n"
        elif "explainer" in node.lower():
            mermaid += f"    style node{i} fill:#e8f5e9,stroke:#1b5e20\\n"
    
    mermaid += "    style START fill:#90caf9,stroke:#0d47a1\\n"
    mermaid += "    style END fill:#a5d6a7,stroke:#2e7d32\\n"
    
    return mermaid


def render_mermaid(mermaid_code: str, height: int = 400) -> None:
    """Render a Mermaid diagram using HTML and JavaScript."""
    # Create HTML with Mermaid rendering
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
        <script>
            mermaid.initialize({{ startOnLoad: true, theme: 'dark', themeVariables: {{ fontSize: '16px' }} }});
        </script>
    </head>
    <body style="background-color: transparent; margin: 0; padding: 20px;">
        <div class="mermaid">
{mermaid_code}
        </div>
    </body>
    </html>
    """
    
    # Render using iframe
    components.html(html_template, height=height, scrolling=True)


def run_migration_workflow(source_file: Path, target_file: Path, mapping_file: Path):
    """Execute the migration workflow with progress tracking"""
    progress_container = st.container()
    status_container = st.container()
    activity_container = st.container()

    with progress_container:
        st.subheader("Migration Progress")

        # Create status placeholders with expandable previews
        load_status = st.empty()
        
        analyst_container = st.container()
        analyst_status = analyst_container.empty()
        analyst_expander = analyst_container.expander("📊 View Schema Analysis", expanded=False)
        
        generator_container = st.container()
        generator_status = generator_container.empty()
        generator_expander = generator_container.expander("📝 View SQL Script", expanded=False)
        
        validator_container = st.container()
        validator_status = validator_container.empty()
        validator_expander = validator_container.expander("🔍 View Validation Results", expanded=False)
        
        explainer_container = st.container()
        explainer_status = explainer_container.empty()
        explainer_expander = explainer_container.expander("📖 View Explanation", expanded=False)

    with activity_container:
        st.subheader("Agent Activity (High-Level)")
        activity_log = st.empty()
        activity_log.info("Waiting to start...")

    def add_event(message: str) -> None:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        st.session_state.events.append({"time": timestamp, "message": message})
    
    def track_execution(node_name: str) -> None:
        """Track which nodes were executed for visualization"""
        st.session_state.execution_path.append(node_name)

    try:
        # Clear previous run data
        st.session_state.events = []
        st.session_state.execution_path = []
        
        # Step 1: Load data
        run_start = time.perf_counter()
        load_start = time.perf_counter()
        load_status.info("📥 Loading input data...")
        activity_log.info("Loading input data and validating schemas...")
        add_event("Load inputs: reading schemas and mappings")

        source_schema = DataLoader.load_schema(str(source_file))
        target_schema = DataLoader.load_schema(str(target_file))
        field_mappings = DataLoader.load_mappings(str(mapping_file))
        input_summary = {
            "source_tables": len(source_schema),
            "target_tables": len(target_schema),
            "field_mappings": len(field_mappings),
        }

        load_status.success(
            f"✅ Loaded {len(source_schema)} source table(s), "
            f"{len(target_schema)} target table(s), "
            f"{len(field_mappings)} field mapping(s)"
        )
        load_end = time.perf_counter()
        add_event("Load inputs: completed")

        # Step 2: Initialize workflow
        analyst_status.info("⚙️ Initializing multi-agent workflow...")
        activity_log.info("Initializing agent graph and shared context...")
        add_event("Initialize: build agent graph and context")
        graph = build_migration_graph()

        context = MigrationContext(
            source_schema=source_schema,
            target_schema=target_schema,
            mappings=field_mappings
        )

        initial_state = MigrationState(
            context=context,
            schema_report=None,
            sql_script=None,
            validation_result=None,
            explanation=None,
            validation_errors=[],
            retry_count=0
        )

        # Step 3: Start Schema Analyst
        analyst_status.info("🔍 Running Schema Analyst Agent...")
        activity_log.info("Schema Analyst: analyzing tables, fields, and mappings...")
        add_event("Schema Analyst: analyzing schema compatibility")

        # Step 4-6: Execute workflow with real-time streaming
        generator_status.info("⏳ Queued: SQL Generator Agent")
        validator_status.info("⏳ Queued: Validation Agent")
        explainer_status.info("⏳ Queued: Explainer Agent")

        # Stream execution to get intermediate results
        graph_start = time.perf_counter()
        final_state = None

        for event in graph.stream(initial_state):
            # Each event is {node_name: state_dict}
            node_name = list(event.keys())[0]
            state_data = event[node_name]
            current_state = MigrationState(**state_data) if isinstance(state_data, dict) else state_data

            if node_name == "analyst":
                track_execution("analyst")
                analyst_status.success("✅ Schema Analyst: Complete")
                activity_log.info("Schema Analyst: completed analysis")
                add_event("Schema Analyst: completed schema compatibility analysis")

                # Update schema preview in expander
                if current_state.schema_report:
                    with analyst_expander:
                        analysis_payload = current_state.schema_report.model_dump()
                        risky = analysis_payload.get("risky_mappings", [])
                        missing = analysis_payload.get("missing_target_fields", [])
                        
                        st.markdown("### Schema Compatibility Analysis")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Risky Mappings", len(risky))
                        with col2:
                            st.metric("Missing Target Fields", len(missing))
                        
                        if risky:
                            st.markdown("**Risky Mappings:**")
                            for mapping in risky[:5]:  # Show first 5
                                st.code(f"{mapping.get('source_field', 'N/A')} → {mapping.get('target_field', 'N/A')}", language='text')
                            if len(risky) > 5:
                                st.info(f"... and {len(risky) - 5} more")
                        
                        if missing:
                            st.markdown("**Missing Target Fields:**")
                            st.code("\n".join(missing[:10]), language='text')  # Show first 10
                            if len(missing) > 10:
                                st.info(f"... and {len(missing) - 10} more")

            elif node_name == "generator":
                track_execution(f"generator (draft #{current_state.retry_count})")
                generator_status.success(f"✅ SQL Generator: Complete (Draft #{current_state.retry_count})")
                activity_log.info("SQL Generator: drafted migration SQL")
                add_event(f"SQL Generator: draft #{current_state.retry_count} generated")

                # Update SQL preview in expander
                with generator_expander:
                    if current_state.sql_script:
                        st.markdown(f"### SQL Migration Script (Draft #{current_state.retry_count})")
                        sql_preview = preview_text(current_state.sql_script, max_lines=30)
                        st.code(sql_preview, language='sql')
                        
                        # Show script stats
                        lines = current_state.sql_script.count('\n') + 1
                        st.caption(f"Script size: {lines} lines, {len(current_state.sql_script)} characters")
                    else:
                        st.info("No SQL script generated")

            elif node_name == "validator":
                if current_state.validation_result and current_state.validation_result.is_valid:
                    track_execution("validator (passed)")
                    validator_status.success("✅ Validation: Passed")
                    add_event("Validation: passed all checks")
                else:
                    track_execution(f"validator (retry #{current_state.retry_count})")
                    validator_status.warning(
                        f"⚠️ Validation: Issues detected (retry #{current_state.retry_count})"
                    )
                    add_event(f"Validation: found issues (retry #{current_state.retry_count})")

                # Update validation preview in expander
                if current_state.validation_result:
                    with validator_expander:
                        validation_payload = current_state.validation_result.model_dump()
                        errors = validation_payload.get("errors", [])
                        warnings = validation_payload.get("warnings", [])
                        
                        st.markdown("### Validation Results")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Errors", len(errors), delta=None if len(errors) == 0 else f"-{len(errors)}")
                        with col2:
                            st.metric("Warnings", len(warnings), delta=None if len(warnings) == 0 else f"-{len(warnings)}")
                        
                        if errors:
                            st.error("**Errors Found:**")
                            for err in errors[:5]:  # Show first 5
                                st.code(err.get('message', str(err)), language='text')
                            if len(errors) > 5:
                                st.info(f"... and {len(errors) - 5} more errors")
                        
                        if warnings:
                            st.warning("**Warnings:**")
                            for warn in warnings[:5]:  # Show first 5
                                st.code(warn.get('message', str(warn)), language='text')
                            if len(warnings) > 5:
                                st.info(f"... and {len(warnings) - 5} more warnings")
                        
                        if not errors and not warnings:
                            st.success("✅ All validation checks passed!")

            elif node_name == "explainer":
                track_execution("explainer")
                explainer_status.success("✅ Explainer: Complete")
                activity_log.success("Explainer: generated human-readable explanation")
                add_event("Explainer: generated explanation")

                # Update explainer preview in expander
                with explainer_expander:
                    if current_state.explanation:
                        st.markdown("### Migration Explanation")
                        st.markdown(current_state.explanation)
                    else:
                        st.info("No explanation generated")

            # Always update final_state to track latest
            final_state = current_state

        graph_end = time.perf_counter()

        # Ensure final_state is available
        if final_state is None:
            raise ValueError("Workflow completed without producing final state")

        # Prepare final outputs
        validation_report = format_validation_report(final_state.validation_result)
        agent_activity = build_agent_activity(final_state)
        analysis_payload = (
            final_state.schema_report.model_dump()
            if final_state.schema_report
            else None
        )
        validation_payload = (
            final_state.validation_result.model_dump()
            if final_state.validation_result
            else None
        )

        # Store outputs in session state
        timing = {
            "load_seconds": round(load_end - load_start, 3),
            "graph_seconds": round(graph_end - graph_start, 3),
            "total_seconds": round(time.perf_counter() - run_start, 3),
            "retries": final_state.retry_count,
        }
        st.session_state.outputs = {
            'sql': final_state.sql_script or "",
            'validation': validation_report,
            'explanation': final_state.explanation or "",
            'analysis': analysis_payload,
            'validation_result': validation_payload,
            'activity': agent_activity,
            'timing': timing,
            'inputs': input_summary,
            'events': st.session_state.events,
            'execution_path': st.session_state.execution_path
        }
        st.session_state.migration_complete = True
        st.session_state.error = None

        # Save outputs to files
        save_start = time.perf_counter()
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)

        with open(output_dir / "sample_output.sql", "w", encoding="utf-8") as f:
            f.write(final_state.sql_script or "")
        with open(output_dir / "validation_report.md", "w", encoding="utf-8") as f:
            f.write(validation_report)
        with open(output_dir / "sql_explanation.md", "w", encoding="utf-8") as f:
            f.write(final_state.explanation or "")
        save_end = time.perf_counter()
        add_event(f"Outputs saved: {round(save_end - save_start, 3)}s")

        with status_container:
            st.success("🎉 Migration orchestration complete! View results below.")

    except Exception as e:
        st.session_state.error = str(e)
        st.session_state.migration_complete = False

        with status_container:
            st.error(f"❌ Migration failed: {str(e)}")
            with st.expander("Error Details"):
                st.code(traceback.format_exc())


def main():
    """Main Streamlit application"""
    init_session_state()
    
    # Warmup embedding model on first app load (async, non-blocking)
    warmup_embedding_model()
    
    # Header
    st.title("🔄 AI Data Migration Orchestrator")
    st.markdown("**Interactive dashboard for schema migration with LLM-powered agents**")
    st.divider()
    
    # Sidebar - File Uploads
    with st.sidebar:
        st.header("📁 Upload Data Files")
        
        source_file = st.file_uploader(
            "Source Schema (JSON)",
            type=['json'],
            help="Upload the source database schema in JSON format"
        )
        
        target_file = st.file_uploader(
            "Target Schema (JSON)",
            type=['json'],
            help="Upload the target database schema in JSON format"
        )
        
        mapping_file = st.file_uploader(
            "Field Mappings (CSV)",
            type=['csv'],
            help=(
                "Upload CSV with columns: SourceTable, SourceColumn, "
                "TargetTable, TargetColumn, TransformationParams"
            )
        )
        
        st.divider()
        
        # Configuration Section
        st.header("⚙️ Configuration")
        
        # Display current LLM provider from environment
        from core.config import get_settings
        settings = get_settings()
        st.info(f"**LLM Provider:** {settings.LLM_PROVIDER}")
        st.caption(f"**Model:** {settings.LLM_MODEL}")
        
        st.divider()
        
        # Run Migration Button
        run_button = st.button(
            "▶️ Run Migration",
            type="primary",
            use_container_width=True,
            disabled=not (source_file and target_file and mapping_file)
        )
        
        if not (source_file and target_file and mapping_file):
            st.warning("⚠️ Please upload all required files to begin")
    
    # Main Content Area
    if run_button:
        # Reset previous state
        st.session_state.migration_complete = False
        st.session_state.outputs = {}
        st.session_state.error = None
        st.session_state.events = []
        
        # Validate files
        st.subheader("📋 Validating Uploads")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            source_data = validate_json_file(source_file)
            if source_data:
                st.success(f"✅ Source schema valid")
                with st.expander("Preview"):
                    st.json(source_data)
        
        with col2:
            target_data = validate_json_file(target_file)
            if target_data:
                st.success(f"✅ Target schema valid")
                with st.expander("Preview"):
                    st.json(target_data)
        
        with col3:
            mapping_valid = validate_csv_file(mapping_file)
            if mapping_valid:
                import pandas as pd
                mapping_file.seek(0)
                df = pd.read_csv(mapping_file)
                st.success(f"✅ Mappings valid ({len(df)} rows)")
                with st.expander("Preview"):
                    st.dataframe(df, width="stretch")
        
        # If all validations pass, save files and run workflow
        if source_data and target_data and mapping_valid:
            st.divider()
            
            # Save uploaded files to data directory
            data_dir = Path("data")
            data_dir.mkdir(exist_ok=True)
            
            source_path = data_dir / "source_schema.json"
            target_path = data_dir / "target_schema.json"
            mapping_path = data_dir / "field_mapping.csv"
            
            save_uploaded_file(source_file, source_path)
            save_uploaded_file(target_file, target_path)
            save_uploaded_file(mapping_file, mapping_path)
            
            # Run the migration workflow
            run_migration_workflow(source_path, target_path, mapping_path)
    
    # Display Results
    if st.session_state.migration_complete and st.session_state.outputs:
        st.divider()
        st.header("📊 Migration Results")
        
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "💾 Generated SQL",
            "✅ Validation Report",
            "📖 SQL Explanation",
            "🔍 Schema Analysis",
            "🧭 Agent Activity",
            "🗺️ Execution Graph"
        ])
        
        with tab1:
            st.subheader("Generated Migration SQL")
            sql_code = st.session_state.outputs.get('sql', '')
            st.code(sql_code, language='sql', line_numbers=True)
            
            st.download_button(
                label="⬇️ Download SQL",
                data=sql_code,
                file_name="migration_output.sql",
                mime="text/sql"
            )
        
        with tab2:
            st.subheader("Validation Report")
            validation_report = st.session_state.outputs.get('validation', '')
            st.markdown(validation_report)
            
            st.download_button(
                label="⬇️ Download Report",
                data=validation_report,
                file_name="validation_report.md",
                mime="text/markdown"
            )
        
        with tab3:
            st.subheader("SQL Explanation")
            explanation = st.session_state.outputs.get('explanation', '')
            st.markdown(explanation)
            
            st.download_button(
                label="⬇️ Download Explanation",
                data=explanation,
                file_name="sql_explanation.md",
                mime="text/markdown"
            )
        
        with tab4:
            st.subheader("Schema Analysis")
            analysis = st.session_state.outputs.get('analysis')
            if analysis:
                st.json(analysis)
            else:
                st.info("No schema analysis available.")

        with tab5:
            st.subheader("Agent Activity (High-Level)")
            activity = st.session_state.outputs.get('activity', {})
            if activity:
                timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
                st.caption(f"Last run: {timestamp}")
                for agent, summary in activity.items():
                    st.write(f"**{agent}:** {summary}")
                inputs = st.session_state.outputs.get('inputs', {})
                if inputs:
                    st.divider()
                    st.write("**Input Summary**")
                    st.json(inputs)
                timing = st.session_state.outputs.get('timing', {})
                if timing:
                    st.divider()
                    st.write("**Timing and Retries**")
                    st.json(timing)
                events = st.session_state.outputs.get('events', [])
                if events:
                    st.divider()
                    st.write("**Activity Timeline**")
                    for event in events:
                        st.write(f"{event['time']} - {event['message']}")
                st.divider()
                st.write("**Agent Previews**")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Schema Analyst**")
                    analysis = st.session_state.outputs.get('analysis')
                    if analysis:
                        risky = analysis.get("risky_mappings", [])
                        missing = analysis.get("missing_target_fields", [])
                        st.write(f"Risky mappings: {len(risky)}")
                        st.write(f"Missing target fields: {len(missing)}")
                        if risky:
                            st.write("Sample risks:")
                            st.json(risky[:3])
                    else:
                        st.info("No schema analysis available.")

                    st.markdown("**Validation Agent**")
                    validation_payload = st.session_state.outputs.get('validation_result')
                    if validation_payload:
                        errors = validation_payload.get("errors", [])
                        warnings = validation_payload.get("warnings", [])
                        st.write(f"Errors: {len(errors)}")
                        st.write(f"Warnings: {len(warnings)}")
                        if errors:
                            st.write("Sample errors:")
                            st.json(errors[:3])
                    else:
                        st.info("No validation results available.")

                with col2:
                    st.markdown("**SQL Generator**")
                    sql_preview = preview_text(
                        st.session_state.outputs.get('sql', ''), max_lines=30
                    )
                    if sql_preview:
                        st.code(sql_preview, language='sql')
                    else:
                        st.info("No SQL output available.")

                    st.markdown("**Explainer Agent**")
                    explanation_preview = preview_text(
                        st.session_state.outputs.get('explanation', ''), max_lines=20
                    )
                    if explanation_preview:
                        st.markdown(explanation_preview)
                    else:
                        st.info("No explanation available.")
            else:
                st.info("No agent activity summary available.")
        
        with tab6:
            st.subheader("Orchestration Graph & Execution Path")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🏗️ Graph Structure")
                st.caption("The complete agent orchestration workflow")
                
                # Generate and display the graph structure
                mermaid_code = generate_graph_mermaid()
                render_mermaid(mermaid_code, height=500)
                
                st.markdown("""
                **Legend:**
                - **Blue**: Schema Analyst
                - **Yellow**: SQL Generator
                - **Purple**: Validator
                - **Green**: Explainer/Success
                - **Red**: Failed validation (max retries)
                
                **Routing Logic:**
                - After validation, if SQL is valid → proceed to Explainer
                - If invalid and retries < 3 → loop back to Generator
                - If invalid and retries >= 3 → end with failure
                """)
            
            with col2:
                st.markdown("### 🛤️ Actual Execution Path")
                st.caption(f"Path taken in this run ({len(st.session_state.execution_path)} nodes)")
                
                if st.session_state.execution_path:
                    # Display execution path as Mermaid diagram
                    execution_mermaid = generate_execution_path_mermaid(st.session_state.execution_path)
                    render_mermaid(execution_mermaid, height=400)
                    
                    # Show step-by-step breakdown
                    st.markdown("**Execution Steps:**")
                    for i, node in enumerate(st.session_state.execution_path, 1):
                        st.write(f"{i}. {node}")
                    
                    # Analyze path
                    retry_count = sum(1 for node in st.session_state.execution_path if "generator" in node.lower())
                    if retry_count > 1:
                        st.warning(f"⚠️ SQL Generator ran {retry_count} times (validation required {retry_count - 1} retries)")
                    else:
                        st.success("✅ SQL passed validation on first attempt!")
                else:
                    st.info("No execution path recorded yet. Run a migration to see the actual path taken.")
    
    # Show error if exists
    elif st.session_state.error:
        st.error(f"Migration failed: {st.session_state.error}")
    
    # Initial state - show instructions
    else:
        st.info("👈 Upload your schemas and field mappings using the sidebar, then click **Run Migration** to begin.")
        
        with st.expander("ℹ️ How to Use"):
            st.markdown("""
            **Step 1:** Upload your source database schema (JSON format)
            
            **Step 2:** Upload your target database schema (JSON format)
            
            **Step 3:** Upload your field mappings (CSV with columns: SourceTable, SourceColumn, TargetTable, TargetColumn, TransformationParams)
            
            **Step 4:** Click the **Run Migration** button
            
            **Step 5:** Monitor the agent workflow progress
            
            **Step 6:** View and download the generated SQL, validation report, and explanation
            """)
        
        with st.expander("🏗️ System Architecture"):
            st.markdown("""
            **Multi-Agent Workflow:**
            1. **Schema Analyst Agent** - Analyzes schemas and mappings
            2. **SQL Generator Agent** - Generates migration SQL using LLM
            3. **Validation Agent** - Validates SQL against target schema
            4. **Explainer Agent** - Creates human-readable documentation
            
            **Powered by:**
            - LangGraph for agent orchestration
            - ChromaDB for semantic memory
            - Redis for workflow checkpoints
            - PostgreSQL for validation testing
            """)
        
        with st.expander("🗺️ View Orchestration Graph"):
            st.markdown("### Agent Workflow Visualization")
            st.caption("This shows how the agents are connected and the routing logic")
            
            mermaid_code = generate_graph_mermaid()
            render_mermaid(mermaid_code, height=450)
            
            st.markdown("""
            **How it works:**
            1. **START** → Schema Analyst analyzes compatibility
            2. Schema Analyst → SQL Generator creates migration script
            3. SQL Generator → Validator checks SQL correctness
            4. Validator decides:
               - ✅ **Valid?** → Explainer creates documentation → **END (Success)**
               - ❌ **Invalid & retries < 3?** → Loop back to SQL Generator with feedback
               - ❌ **Invalid & retries >= 3?** → **END (Failure)**
            
            **Smart Retry Loop:**
            The validator provides specific error feedback to the generator, allowing it to fix issues automatically!
            """)


if __name__ == "__main__":
    main()
