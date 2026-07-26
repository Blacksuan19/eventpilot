"""Run the EventPilot dashboard and autonomous supervisor."""

import argparse
import asyncio
from time import monotonic, time

import uvicorn
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from eventpilot.adapters.adaptyv.mock import MockFoundryClient
from eventpilot.config import Settings, get_settings
from eventpilot.core.agent_reasoning import (
    AutonomousReasoningEngine,
    InstructorAutonomousReasoningEngine,
)
from eventpilot.core.autonomous import (
    AgentRuntime,
    build_autonomous_graph,
)
from eventpilot.core.clock import AcceleratedClock
from eventpilot.core.reporting import (
    AgentReporter,
    CompositeAgentReporter,
    ConsoleAgentReporter,
)
from eventpilot.dashboard.app import DashboardEventStore, create_dashboard_app
from eventpilot.dashboard.supervisor import AgentTaskSupervisor
from eventpilot.notifications.console import ConsoleNotificationSink
from eventpilot.sources.adaptyv import AdaptyvDataSource
from eventpilot.sources.adaptyv_demo import DemoAdaptyvReasoningEngine
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


def _create_runtime(
    settings: Settings,
    checkpointer: AsyncSqliteSaver,
    *,
    reporter: AgentReporter | None = None,
) -> AgentRuntime:
    """Build one graph runtime around shared durable checkpoint infrastructure."""
    max_tool_calls_per_cycle = 32
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
    graph = build_autonomous_graph(
        agent,
        source,
        ConsoleNotificationSink(),
        destination=settings.notification_destination,
        checkpointer=checkpointer,
        sleep=accelerated_clock.sleep if accelerated_clock else asyncio.sleep,
        idle_sleep=(accelerated_clock.sleep_unbounded if accelerated_clock else asyncio.sleep),
        clock=accelerated_clock or time,
        max_tool_calls_per_cycle=max_tool_calls_per_cycle,
        external_call_timeout_seconds=settings.external_call_timeout_seconds,
        retry_max_attempts=settings.retry_max_attempts,
        retry_initial_interval_seconds=settings.retry_initial_interval_seconds,
        reporter=reporter,
    )
    return AgentRuntime(graph)


async def run_dashboard() -> None:
    """Run the autonomous supervisor beside its live presentation dashboard."""
    settings = get_settings()
    store = DashboardEventStore(path=settings.database_path.with_suffix(".events.jsonl"))
    reporter = CompositeAgentReporter(ConsoleAgentReporter(), store)

    settings.database_path.parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(str(settings.database_path)) as checkpointer:

        async def reset_state() -> None:
            """Clear graph checkpoints and dashboard events before a fresh task starts."""
            await checkpointer.adelete_thread(AgentRuntime.thread_id)
            store.clear()

        supervisor = AgentTaskSupervisor(
            lambda: _create_runtime(settings, checkpointer, reporter=reporter),
            reset_state,
        )
        server = uvicorn.Server(
            uvicorn.Config(
                create_dashboard_app(
                    store,
                    reset_agent=supervisor.reset,
                    resolve_approval=supervisor.resolve_approval,
                    get_agent_health=supervisor.health,
                ),
                host=settings.dashboard_host,
                port=settings.dashboard_port,
                log_level="warning",
            )
        )
        supervisor.start()
        try:
            await server.serve()
        finally:
            await supervisor.stop()


def main() -> None:
    """Start the dashboard and its autonomous EventPilot supervisor."""
    parser = argparse.ArgumentParser(description="EventPilot autonomous operations agent")
    parser.add_argument("command", choices=["dashboard"])
    parser.parse_args()
    asyncio.run(run_dashboard())
