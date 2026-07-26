"""LangGraph topology for the autonomous agent supervisor."""

import asyncio
from collections.abc import Callable
from time import time
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from eventpilot.core.agent_reasoning import AutonomousReasoningEngine
from eventpilot.core.autonomous.nodes import AutonomousGraphNodes
from eventpilot.core.autonomous.state import AutonomousAgentState, Sleep
from eventpilot.core.reporting import AgentReporter, ConsoleAgentReporter
from eventpilot.notifications.base import NotificationSink
from eventpilot.sources.base import DataSource


def build_autonomous_graph(
    agent: AutonomousReasoningEngine,
    source: DataSource,
    sink: NotificationSink,
    *,
    max_wait_seconds: int,
    destination: str = "local-console",
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    sleep: Sleep = asyncio.sleep,
    idle_sleep: Sleep | None = None,
    clock: Callable[[], float] = time,
    external_call_timeout_seconds: float = 60.0,
    retry_max_attempts: int = 3,
    retry_initial_interval_seconds: float = 0.5,
    reporter: AgentReporter | None = None,
) -> Any:
    """Build a generic supervisor around a registered monitoring data source."""
    nodes = AutonomousGraphNodes(
        agent,
        source,
        sink,
        destination=destination,
        sleep=sleep,
        idle_sleep=idle_sleep,
        clock=clock,
        max_wait_seconds=max_wait_seconds,
        reporter=reporter or ConsoleAgentReporter(),
    )

    retry_policy = RetryPolicy(
        initial_interval=retry_initial_interval_seconds,
        max_attempts=retry_max_attempts,
    )
    builder = StateGraph(AutonomousAgentState)
    builder.add_node(
        "agent",
        nodes.reason,
        retry_policy=retry_policy,
        timeout=external_call_timeout_seconds,
    )
    builder.add_node(
        "source_tool",
        nodes.execute_source_tool,
        timeout=external_call_timeout_seconds,
    )
    builder.add_node(
        "retryable_source_tool",
        nodes.execute_source_tool,
        retry_policy=retry_policy,
        timeout=external_call_timeout_seconds,
    )
    builder.add_node(
        "parallel_source_tool",
        nodes.execute_parallel_source_tool,
        retry_policy=retry_policy,
        timeout=external_call_timeout_seconds,
    )
    builder.add_node("reduce_parallel_source_tools", nodes.reduce_parallel_source_tools)
    builder.add_node("select_objective", nodes.choose_objective)
    builder.add_node(
        "request_approval",
        nodes.request_approval,
        timeout=external_call_timeout_seconds,
    )
    builder.add_node("publish_approval", nodes.publish_approval)
    builder.add_node("human_approval", nodes.human_approval)
    builder.add_node("reject_approval", nodes.reject_approval)
    builder.add_node(
        "send_alert",
        nodes.send_alert,
        timeout=external_call_timeout_seconds,
    )
    builder.add_node("prepare_wait", nodes.prepare_wait)
    builder.add_node("wait", nodes.wait)
    builder.add_node("end_invocation", nodes.end_invocation)
    builder.add_node("reject_action", nodes.reject_action)

    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        nodes.route_tool,
        {
            "source_tool": "source_tool",
            "retryable_source_tool": "retryable_source_tool",
            "select_objective": "select_objective",
            "request_approval": "request_approval",
            "send_alert": "send_alert",
            "wait": "prepare_wait",
            "reject_action": "reject_action",
        },
    )
    builder.add_edge("request_approval", "publish_approval")
    builder.add_edge("publish_approval", "human_approval")
    builder.add_edge("source_tool", "agent")
    builder.add_edge("retryable_source_tool", "agent")
    builder.add_edge("parallel_source_tool", "reduce_parallel_source_tools")
    builder.add_edge("reduce_parallel_source_tools", "agent")
    builder.add_edge("select_objective", "agent")
    builder.add_edge("reject_approval", "agent")
    builder.add_conditional_edges(
        "prepare_wait",
        nodes.route_prepared_wait,
        {"wait": "wait", "agent": "agent"},
    )
    builder.add_conditional_edges(
        "wait",
        nodes.route_invocation_end,
        {"end": "end_invocation", "agent": "agent"},
    )
    builder.add_edge("reject_action", "agent")
    builder.add_conditional_edges(
        "send_alert",
        nodes.route_invocation_end,
        {"end": "end_invocation", "agent": "agent"},
    )
    builder.add_edge("end_invocation", END)
    return builder.compile(checkpointer=checkpointer or InMemorySaver())
