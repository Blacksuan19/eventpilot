"""LangGraph topology for the autonomous agent supervisor."""

import asyncio
from collections.abc import Callable
from time import time
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

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
    destination: str = "local-console",
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    sleep: Sleep = asyncio.sleep,
    idle_sleep: Sleep | None = None,
    clock: Callable[[], float] = time,
    max_wait_seconds: int | None = None,
    max_tool_calls_per_cycle: int = 32,
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
        max_tool_calls_per_cycle=max_tool_calls_per_cycle,
        reporter=reporter or ConsoleAgentReporter(),
    )

    builder = StateGraph(AutonomousAgentState)
    builder.add_node("agent", nodes.reason)
    builder.add_node("source_tool", nodes.execute_source_tool)
    builder.add_node("select_objective", nodes.choose_objective)
    builder.add_node("request_approval", nodes.request_approval)
    builder.add_node("human_approval", nodes.human_approval)
    builder.add_node("reject_approval", nodes.reject_approval)
    builder.add_node("send_alert", nodes.send_alert)
    builder.add_node("wait", nodes.wait)
    builder.add_node("finish_cycle", nodes.finish_cycle)
    builder.add_node("reject_action", nodes.reject_action)

    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        nodes.route_tool,
        {
            "source_tool": "source_tool",
            "select_objective": "select_objective",
            "request_approval": "request_approval",
            "send_alert": "send_alert",
            "wait": "wait",
            "finish_cycle": "finish_cycle",
            "reject_action": "reject_action",
        },
    )
    builder.add_edge("request_approval", "human_approval")
    builder.add_edge("source_tool", "agent")
    builder.add_edge("select_objective", "agent")
    builder.add_edge("reject_approval", "agent")
    builder.add_edge("wait", "agent")
    builder.add_edge("reject_action", "agent")
    builder.add_conditional_edges(
        "send_alert",
        nodes.route_cycle_end,
        {"end": END, "agent": "agent"},
    )
    builder.add_conditional_edges(
        "finish_cycle",
        nodes.route_cycle_end,
        {"end": END, "agent": "agent"},
    )
    return builder.compile(checkpointer=checkpointer or InMemorySaver())
