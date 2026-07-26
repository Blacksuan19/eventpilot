"""Verify CLI factories compose configured runtime dependencies."""

import asyncio
from pathlib import Path
from typing import Any, ClassVar, cast

import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from eventpilot import cli
from eventpilot.adapters.adaptyv.mock import MockFoundryClient
from eventpilot.config import Settings
from eventpilot.core.agent_reasoning import AgentTurn
from eventpilot.core.reporting import AgentEvent, AgentReporter
from eventpilot.sources.adaptyv import AdaptyvDataSource
from eventpilot.sources.adaptyv_demo import DemoAdaptyvReasoningEngine


class RecordingInstructorEngine:
    """Capture the provider configuration supplied by the CLI factory."""

    created_with: ClassVar[dict[str, Any]] = {}

    def __init__(self, model: str, source: Any, **options: Any) -> None:
        """Store constructor inputs for assertions."""
        type(self).created_with = {"model": model, "source": source, **options}

    async def decide(
        self, transcript: list[dict[str, Any]], source_state: dict[str, Any]
    ) -> AgentTurn:
        """Satisfy the reasoning protocol without participating in composition tests."""
        raise AssertionError("Composition tests must not run the reasoning engine")


class RecordingClock:
    """Capture accelerated-clock construction and expose its runtime methods."""

    instance: "RecordingClock | None" = None

    def __init__(self, acceleration: float, *, max_physical_wait_seconds: float) -> None:
        """Record both configured timing controls."""
        self.acceleration = acceleration
        self.max_physical_wait_seconds = max_physical_wait_seconds
        type(self).instance = self

    def __call__(self) -> float:
        """Return a stable logical timestamp."""
        return 100.0

    async def sleep(self, seconds: float) -> None:
        """Provide the active-wait callable expected by graph composition."""

    async def sleep_unbounded(self, seconds: float) -> None:
        """Provide the idle-wait callable expected by graph composition."""


class NoopReporter:
    """Provide a concrete reporter dependency for graph composition."""

    def emit(self, event: AgentEvent) -> None:
        """Accept an event without changing composition state."""


class RecordingRuntime:
    """Capture the compiled graph and recursion policy."""

    def __init__(self, graph: Any, *, recursion_limit: int) -> None:
        """Store runtime constructor inputs."""
        self.graph = graph
        self.recursion_limit = recursion_limit


def test_reasoning_factory_selects_the_explicit_demo_engine() -> None:
    """Use the deterministic engine only when mock mode is configured."""
    source = AdaptyvDataSource(MockFoundryClient.from_fixture())

    engine = cli._build_reasoning_engine(Settings(mock_llm=True), source)

    assert isinstance(engine, DemoAdaptyvReasoningEngine)


def test_reasoning_factory_passes_llm_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forward provider credentials and wait policy into Instructor."""
    monkeypatch.setattr(cli, "InstructorAutonomousReasoningEngine", RecordingInstructorEngine)
    source = AdaptyvDataSource(MockFoundryClient.from_fixture())
    settings = Settings.model_validate(
        {
            "mock_llm": False,
            "LLM_PROVIDER": "openai",
            "LLM_MODEL": "gpt-test",
            "LLM_API_KEY": "secret",
            "LLM_API_BASE": "https://llm.example/v1",
            "max_wait_seconds": 900,
        }
    )

    engine = cli._build_reasoning_engine(settings, source)

    assert isinstance(engine, RecordingInstructorEngine)
    assert RecordingInstructorEngine.created_with == {
        "model": "openai/gpt-test",
        "source": source,
        "api_key": "secret",
        "api_base": "https://llm.example/v1",
        "max_wait_seconds": 900,
    }


def test_reasoning_factory_rejects_missing_llm_configuration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Fail startup clearly when neither a live model nor mock mode is configured."""
    for name in ("LLM_PROVIDER", "LLM_MODEL", "LLM_API_KEY", "LLM_API_BASE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    source = AdaptyvDataSource(MockFoundryClient.from_fixture())
    settings = Settings(mock_llm=False)

    with pytest.raises(RuntimeError, match="Configure LLM_PROVIDER and LLM_MODEL"):
        cli._build_reasoning_engine(settings, source)


def test_runtime_factory_wires_accelerated_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pass every configured runtime policy to the graph and process loop."""
    graph = object()
    captured: dict[str, Any] = {}

    def record_graph(agent: Any, source: Any, sink: Any, **options: Any) -> object:
        """Capture graph dependencies without compiling LangGraph."""
        captured.update(agent=agent, source=source, sink=sink, **options)
        return graph

    monkeypatch.setattr(cli, "AcceleratedClock", RecordingClock)
    monkeypatch.setattr(cli, "build_autonomous_graph", record_graph)
    monkeypatch.setattr(cli, "AgentRuntime", RecordingRuntime)
    settings = Settings(
        mock_llm=True,
        notification_destination="ops-room",
        time_acceleration=60,
        max_physical_wait_seconds=2,
        max_wait_seconds=900,
        recursion_limit=77,
        external_call_timeout_seconds=12,
        retry_max_attempts=5,
        retry_initial_interval_seconds=0.25,
    )
    checkpointer = cast(AsyncSqliteSaver, object())
    reporter = cast(AgentReporter, NoopReporter())

    runtime = cast(Any, cli._create_runtime(settings, checkpointer, reporter=reporter))

    clock = RecordingClock.instance
    assert clock is not None
    assert (clock.acceleration, clock.max_physical_wait_seconds) == (60, 2)
    assert isinstance(captured["agent"], DemoAdaptyvReasoningEngine)
    assert isinstance(captured["source"], AdaptyvDataSource)
    assert captured["destination"] == "ops-room"
    assert captured["checkpointer"] is checkpointer
    assert captured["clock"] is clock
    assert captured["sleep"].__self__ is clock
    assert captured["idle_sleep"].__self__ is clock
    assert captured["max_wait_seconds"] == 900
    assert captured["external_call_timeout_seconds"] == 12
    assert captured["retry_max_attempts"] == 5
    assert captured["retry_initial_interval_seconds"] == 0.25
    assert captured["reporter"] is reporter
    assert runtime.graph is graph
    assert runtime.recursion_limit == 77


def test_runtime_factory_uses_real_time_without_acceleration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compose normal asyncio waits when time compression is disabled."""
    captured: dict[str, Any] = {}

    def record_graph(agent: Any, source: Any, sink: Any, **options: Any) -> object:
        """Capture the unaccelerated graph timing dependencies."""
        captured.update(options)
        return object()

    monkeypatch.setattr(cli, "build_autonomous_graph", record_graph)
    monkeypatch.setattr(cli, "AgentRuntime", RecordingRuntime)

    cli._create_runtime(
        Settings(mock_llm=True, time_acceleration=1),
        cast(AsyncSqliteSaver, object()),
    )

    assert captured["sleep"] is asyncio.sleep
    assert captured["idle_sleep"] is asyncio.sleep
    assert captured["clock"] is cli.time
