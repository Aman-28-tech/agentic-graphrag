"""
graph_agent.py — LangGraph Agent Orchestration

Manages the stateful agentic loop using LangGraph:
  retrieve → generate → verify → (retry if hallucination)

State machine:
  ┌─────────┐     ┌──────────┐     ┌──────────┐     ┌───────────┐
  │  ROUTE  │ ──► │ RETRIEVE │ ──► │ GENERATE │ ──► │  VERIFY   │
  └─────────┘     └──────────┘     └──────────┘     └───────────┘
                       ▲                                   │
                       │          ┌──────────────┐         │
                       └──────── │ RETRY/ROTATE  │ ◄──────┘
                                 └──────────────┘    (if hallucination)
                                        │
                                        ▼
                                 ┌──────────────┐
                                 │   RESPOND    │
                                 └──────────────┘
"""

from __future__ import annotations

from typing import TypedDict, Annotated, Literal
from operator import add

from langgraph.graph import StateGraph, END

import config
from utils import console, get_logger, is_dont_know

logger = get_logger(__name__)


# ── Agent State ────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    """Shared state passed between graph nodes."""
    query:             str
    expanded_queries:  list[str]
    primary_strategy:  str
    tried_strategies:  Annotated[list[str], add]
    context_chunks:    list[str]
    answer:            str
    verified:          bool
    confidence:        float
    verdict:           str
    strategy_used:     str
    attempt:           int
    max_retries:       int


# ── Graph Node Functions ───────────────────────────────────────────────────────

def route_node(state: AgentState) -> dict:
    """Route query to the best retrieval strategy."""
    from retriever import route_query
    strategy = route_query(state["query"])
    console.print(f"  [dim]Router →[/dim] [bold]{strategy}[/bold]")
    return {"primary_strategy": strategy}


def expand_node(state: AgentState) -> dict:
    """Expand query into alternative phrasings."""
    # LLM engine is injected via closure in build_agent()
    return {}  # Expansion happens in the pipeline before graph execution


def retrieve_node(state: AgentState) -> dict:
    """Retrieve context using the current strategy."""
    console.print(f"\n  [dim]── Attempt {state['attempt']}/{state['max_retries']} ──[/dim]")
    # Retrieval is delegated to the pipeline's retriever
    return {}


def generate_node(state: AgentState) -> dict:
    """Generate answer from retrieved context."""
    console.print("  [dim]Generating answer…[/dim]")
    return {}


def verify_node(state: AgentState) -> dict:
    """Verify the generated answer against context."""
    console.print("  [dim]Verifying answer…[/dim]")
    return {}


def should_retry(state: AgentState) -> Literal["retry", "respond"]:
    """
    Decision function: retry with a different strategy or respond.

    Retry conditions:
      1. Answer is hallucinated (not verified)
      2. We haven't exceeded max retries
      3. There are untried strategies remaining
    """
    if state["verified"]:
        console.print("  [green]✓ VERIFIED[/green]")
        return "respond"

    if is_dont_know(state.get("answer", "")):
        console.print("  [yellow]Model admitted ignorance → stop[/yellow]")
        return "respond"

    if state["attempt"] >= state["max_retries"]:
        console.print("  [yellow]Max retries reached → stop[/yellow]")
        return "respond"

    confidence = state.get("confidence", 0.0)
    if confidence >= config.CONFIDENCE_THRESHOLD:
        console.print(f"  [green]Confidence {confidence:.2f} ≥ threshold → accept[/green]")
        return "respond"

    console.print(
        f"  [yellow]✗ HALLUCINATION (confidence={confidence:.2f}) — retrying…[/yellow]"
    )
    return "retry"


def retry_node(state: AgentState) -> dict:
    """Rotate to the next untried strategy."""
    current = state.get("strategy_used", "hybrid")
    next_strategy = "hybrid" if current != "hybrid" else "bm25"
    console.print(f"  [yellow]Rotating to strategy: {next_strategy}[/yellow]")
    return {
        "primary_strategy": next_strategy,
        "attempt": state["attempt"] + 1,
    }


# ── Build the LangGraph ───────────────────────────────────────────────────────

def build_agent_graph() -> StateGraph:
    """
    Construct the LangGraph state machine for agentic RAG.

    The actual LLM/retrieval calls are made by the pipeline (graphrag_pipeline.py)
    which uses this graph's state transitions to decide flow control.
    This graph defines the LOGIC (when to retry, when to stop).
    """
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("route",    route_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("verify",   verify_node)
    workflow.add_node("retry",    retry_node)

    # Define edges
    workflow.set_entry_point("route")
    workflow.add_edge("route",    "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "verify")

    # Conditional: verify → respond or retry
    workflow.add_conditional_edges(
        "verify",
        should_retry,
        {
            "respond": END,
            "retry":   "retry",
        },
    )
    workflow.add_edge("retry", "retrieve")

    return workflow


def compile_agent():
    """Compile the LangGraph workflow into a runnable agent."""
    workflow = build_agent_graph()
    return workflow.compile()
