"""Run EventPilot as a bounded demonstration or continuous supervisor."""

import argparse
import asyncio
from contextlib import suppress
from pathlib import Path
from time import monotonic, time

import uvicorn
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from eventpilot.adapters.adaptyv.mock import MockFoundryClient
from eventpilot.config import Settings, get_settings
from eventpilot.core.agent_reasoning import (
    AutonomousReasoningEngine,
    InstructorAutonomousReasoningEngine,
)
from eventpilot.core.autonomous import AgentRuntime, build_autonomous_graph
from eventpilot.core.clock import AcceleratedClock
from eventpilot.core.reporting import (
    AgentReporter,
    CompositeAgentReporter,
    ConsoleAgentReporter,
)
from eventpilot.dashboard.app import DashboardEventStore, create_dashboard_app
from eventpilot.notifications.console import ConsoleNotificationSink
from eventpilot.sources.adaptyv import AdaptyvDataSource, DemoAdaptyvReasoningEngine
from eventpilot.sources.base import DataSource


def _build_reasoning_engine(
    settings: Settings, source: DataSource, *, max_tool_calls_per_cycle: int = 32
) -> AutonomousReasoningEngine:
    """Create either the configured LLM agent or the explicit offline test double."""
    if settings.mock_llm:
        return DemoAdaptyvReasoningEngine()
    if not settings.instructor_model:
        raise RuntimeError("Configure LLM_PROVIDER and LLM_MODEL or set EVENTPILOT_MOCK_LLM=true")
    return InstructorAutonomousReasoningEngine(
        settings.instructor_model,
        source,
        api_key=(settings.llm_api_key.get_secret_value() if settings.llm_api_key else None),
        api_base=settings.llm_api_base,
        max_tool_calls_per_cycle=max_tool_calls_per_cycle,
    )


async def run_agent(
    *, max_cycles: int | None = None, reporter: AgentReporter | None = None
) -> None:
    """Run the autonomous supervisor continuously or for a bounded local demonstration."""
    settings = get_settings()
    settings.database_path.parent.mkdir(parents=True, exist_ok=True)
    max_tool_calls_per_cycle = 12 if max_cycles is not None else 32
    accelerated_clock = (
        AcceleratedClock(
            settings.time_acceleration,
            max_physical_wait_seconds=settings.max_physical_wait_seconds,
        )
        if settings.time_acceleration > 1
        else None
    )
    foundry = MockFoundryClient.from_fixture(clock=accelerated_clock or monotonic)
    source = AdaptyvDataSource(foundry)
    agent = _build_reasoning_engine(
        settings, source, max_tool_calls_per_cycle=max_tool_calls_per_cycle
    )
    async with AsyncSqliteSaver.from_conn_string(str(settings.database_path)) as checkpointer:
        graph = build_autonomous_graph(
            agent,
            source,
            ConsoleNotificationSink(),
            destination=settings.notification_destination,
            checkpointer=checkpointer,
            sleep=accelerated_clock.sleep if accelerated_clock else asyncio.sleep,
            idle_sleep=(accelerated_clock.sleep_unbounded if accelerated_clock else asyncio.sleep),
            clock=accelerated_clock or time,
            max_wait_seconds=(2 if max_cycles is not None and not accelerated_clock else None),
            max_tool_calls_per_cycle=max_tool_calls_per_cycle,
            reporter=reporter,
        )
        runtime = AgentRuntime(graph)
        print("EventPilot autonomous supervisor started")
        final = await runtime.run(max_cycles=max_cycles)
        if max_cycles is not None:
            print(f"Cycle complete: {final.get('cycle_summary')}")


async def run_dashboard() -> None:
    """Run the autonomous supervisor beside its live presentation dashboard."""
    settings = get_settings()
    store = DashboardEventStore(path=settings.database_path.with_suffix(".events.jsonl"))
    reporter = CompositeAgentReporter(ConsoleAgentReporter(), store)
    reset_lock = asyncio.Lock()
    agent_task: asyncio.Task[None] | None = None

    def start_agent() -> asyncio.Task[None]:
        """Start one supervisor task using the shared dashboard reporter."""
        return asyncio.create_task(run_agent(reporter=reporter), name="eventpilot-agent")

    async def reset_agent() -> None:
        """Cancel the supervisor, clear durable state, and start a fresh run."""
        nonlocal agent_task
        async with reset_lock:
            if agent_task and not agent_task.done():
                agent_task.cancel()
                with suppress(asyncio.CancelledError):
                    await agent_task
            for checkpoint_file in (
                settings.database_path,
                Path(f"{settings.database_path}-shm"),
                Path(f"{settings.database_path}-wal"),
            ):
                checkpoint_file.unlink(missing_ok=True)
            store.clear()
            agent_task = start_agent()

    server = uvicorn.Server(
        uvicorn.Config(
            create_dashboard_app(store, reset_agent=reset_agent),
            host=settings.dashboard_host,
            port=settings.dashboard_port,
            log_level="warning",
        )
    )
    agent_task = start_agent()
    try:
        await server.serve()
        if not agent_task.done():
            agent_task.cancel()
        with suppress(asyncio.CancelledError):
            await agent_task
    finally:
        if not agent_task.done():
            agent_task.cancel()


def main() -> None:
    """Parse the command line and start the autonomous EventPilot runtime."""
    parser = argparse.ArgumentParser(description="EventPilot autonomous experiment agent")
    parser.add_argument("command", choices=["run", "demo", "dashboard"])
    args = parser.parse_args()
    if args.command == "dashboard":
        asyncio.run(run_dashboard())
    else:
        asyncio.run(run_agent(max_cycles=1 if args.command == "demo" else None))
