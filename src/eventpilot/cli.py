"""Run EventPilot as a bounded demonstration or continuous supervisor."""

import argparse
import asyncio

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from eventpilot.config import Settings, get_settings
from eventpilot.core.agent_reasoning import (
    AutonomousReasoningEngine,
    DemoAutonomousReasoningEngine,
    InstructorAutonomousReasoningEngine,
)
from eventpilot.core.autonomous import AgentRuntime, build_autonomous_graph
from eventpilot.notifications.console import ConsoleNotificationSink
from eventpilot.simulator import MockFoundryClient


def _build_reasoning_engine(
    settings: Settings, *, max_tool_calls_per_cycle: int = 32
) -> AutonomousReasoningEngine:
    """Create either the configured LLM agent or the explicit offline test double."""
    if settings.mock_llm:
        return DemoAutonomousReasoningEngine()
    if not settings.instructor_model:
        raise RuntimeError("Configure LLM_PROVIDER and LLM_MODEL or set EVENTPILOT_MOCK_LLM=true")
    return InstructorAutonomousReasoningEngine(
        settings.instructor_model,
        api_key=(settings.llm_api_key.get_secret_value() if settings.llm_api_key else None),
        api_base=settings.llm_api_base,
        max_tool_calls_per_cycle=max_tool_calls_per_cycle,
    )


async def run_agent(*, max_cycles: int | None = None) -> None:
    """Run the autonomous supervisor continuously or for a bounded local demonstration."""
    settings = get_settings()
    settings.database_path.parent.mkdir(parents=True, exist_ok=True)
    agent = _build_reasoning_engine(
        settings, max_tool_calls_per_cycle=12 if max_cycles is not None else 32
    )
    foundry = MockFoundryClient.from_fixture()
    async with AsyncSqliteSaver.from_conn_string(str(settings.database_path)) as checkpointer:
        graph = build_autonomous_graph(
            agent,
            foundry,
            ConsoleNotificationSink(),
            destination=settings.notification_destination,
            checkpointer=checkpointer,
            max_wait_seconds=2 if max_cycles is not None else None,
        )
        runtime = AgentRuntime(graph)
        print("EventPilot autonomous supervisor started")
        final = await runtime.run(max_cycles=max_cycles)
        if max_cycles is not None:
            print(f"Cycle complete: {final.get('cycle_summary')}")


def main() -> None:
    """Parse the command line and start the autonomous EventPilot runtime."""
    parser = argparse.ArgumentParser(description="EventPilot autonomous experiment agent")
    parser.add_argument("command", choices=["run", "demo"])
    args = parser.parse_args()
    asyncio.run(run_agent(max_cycles=1 if args.command == "demo" else None))
