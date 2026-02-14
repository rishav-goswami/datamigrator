from __future__ import annotations

from typing import Literal

from langgraph.graph import END, START, StateGraph

from agents.analyst import SchemaAnalystAgent
from agents.explainer import ExplainerAgent
from agents.sql_generator import SQLGeneratorAgent
from agents.validator import ValidationAgent
from core.config import get_settings
from core.llm import LLMClient
from core.memory import MemoryClient
from core.models import ValidationError, ValidationResult
from core.state import MigrationState


def analyst_node(state: MigrationState) -> MigrationState:
    llm = LLMClient().get_client()
    memory = MemoryClient()
    agent = SchemaAnalystAgent(llm, memory)

    state.schema_report = agent.analyze(state.context)
    return state


def sql_generator_node(state: MigrationState) -> MigrationState:
    llm = LLMClient().get_client()
    agent = SQLGeneratorAgent(llm)

    feedback = state.validation_errors if state.retry_count > 0 else None

    result = agent.generate(
        state.schema_report,
        state.context,
        feedback,
        retry_count=state.retry_count,
    )

    state.sql_script = result.sql_script
    state.retry_count += 1
    return state


def validator_node(state: MigrationState) -> MigrationState:
    if state.sql_script is None or state.schema_report is None:
        state.validation_result = ValidationResult(
            is_valid=False,
            errors=[
                ValidationError(
                    severity="ERROR",
                    category="runtime",
                    message="Missing schema report or SQL script from previous steps.",
                )
            ],
        )
        state.validation_errors = [e.message for e in state.validation_result.errors]
        return state

    settings = get_settings()
    agent = ValidationAgent(settings.DATABASE_URL)

    result = agent.validate(state.sql_script, state.schema_report, state.context)

    state.validation_result = result
    state.validation_errors = [e.message for e in result.errors]
    return state


def explainer_node(state: MigrationState) -> MigrationState:
    if (
        state.sql_script is None
        or state.schema_report is None
        or state.validation_result is None
    ):
        state.explanation = "# Migration Explanation\n\nCould not generate: missing inputs from previous steps."
        return state

    llm = LLMClient().get_client()
    agent = ExplainerAgent(llm)

    state.explanation = agent.explain(
        state.sql_script, state.schema_report, state.validation_result, state.context
    )
    return state


def should_retry_or_explain(
    state: MigrationState,
) -> Literal["regenerate", "explain", "fail"]:
    if not state.validation_result.is_valid:
        if state.retry_count < 3:
            return "regenerate"
        else:
            return "fail"
    return "explain"


def build_migration_graph() -> StateGraph:
    graph = StateGraph(MigrationState)

    graph.add_node("analyst", analyst_node)
    graph.add_node("generator", sql_generator_node)
    graph.add_node("validator", validator_node)
    graph.add_node("explainer", explainer_node)

    graph.add_edge(START, "analyst")
    graph.add_edge("analyst", "generator")
    graph.add_edge("generator", "validator")

    graph.add_conditional_edges(
        "validator",
        should_retry_or_explain,
        {
            "regenerate": "generator",
            "explain": "explainer",
            "fail": END,
        },
    )

    graph.add_edge("explainer", END)

    return graph.compile()
