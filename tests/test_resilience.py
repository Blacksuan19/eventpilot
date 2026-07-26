"""Verify retry and timeout policy at autonomous graph boundaries."""

import asyncio
from typing import Any, Literal

import pytest
from langgraph.errors import NodeTimeoutError

from eventpilot.adapters.adaptyv.mock import MockFoundryClient
from eventpilot.adapters.adaptyv.models import ExperimentPage
from eventpilot.adapters.adaptyv.tools import ListExperiments
from eventpilot.core.agent_reasoning import AgentTurn, FinishCycle
from eventpilot.core.autonomous import AgentRuntime, build_autonomous_graph
from eventpilot.core.notifications import DeliveryResult, Notification
from eventpilot.sources.adaptyv import AdaptyvDataSource
from eventpilot.sources.base import SourceContext, SourceExecution, SourceToolCall


class FlakyFinishAgent:
    """Fail one model request before returning a valid terminal decision."""

    def __init__(self) -> None:
        """Start with no attempted model calls."""
        self.attempts = 0

    async def decide(
        self, transcript: list[dict[str, Any]], source_state: dict[str, Any]
    ) -> AgentTurn:
        """Raise a transient connection error once, then finish the cycle."""
        self.attempts += 1
        if self.attempts == 1:
            raise ConnectionError("temporary model connection failure")
        return AgentTurn(
            rationale="The transient model failure recovered.",
            action=FinishCycle(summary="Recovered model call."),
        )


class HangingAgent:
    """Model an external reasoning request that never returns promptly."""

    async def decide(
        self, transcript: list[dict[str, Any]], source_state: dict[str, Any]
    ) -> AgentTurn:
        """Sleep long enough for the graph's node timeout to cancel the request."""
        await asyncio.sleep(60)
        raise AssertionError("The timeout should cancel this call")


class DiscoveryThenFinishAgent:
    """Discover source resources and then end the current cycle."""

    def __init__(self) -> None:
        """Create the fixed two-turn trajectory."""
        self._turns = iter(
            [
                AgentTurn(rationale="Discover resources.", action=ListExperiments()),
                AgentTurn(
                    rationale="Discovery recovered and completed.",
                    action=FinishCycle(summary="Recovered source call."),
                ),
            ]
        )

    async def decide(
        self, transcript: list[dict[str, Any]], source_state: dict[str, Any]
    ) -> AgentTurn:
        """Return the next deterministic test decision."""
        return next(self._turns)


class FlakyListFoundryClient(MockFoundryClient):
    """Fail one repeatable Foundry read before serving fixture data."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize fixture state and the read attempt counter."""
        super().__init__(*args, **kwargs)
        self.list_attempts = 0

    async def list_experiments(self, *, limit: int = 50, offset: int = 0) -> ExperimentPage:
        """Raise one transient connection error before delegating to the fixture client."""
        self.list_attempts += 1
        if self.list_attempts == 1:
            raise ConnectionError("temporary source connection failure")
        return await super().list_experiments(limit=limit, offset=offset)


class NoopSink:
    """Satisfy the graph notification contract without external delivery."""

    channel_name = "noop"

    async def send(
        self,
        destination: str,
        notification: Notification,
        *,
        idempotency_key: str,
    ) -> DeliveryResult:
        """Return a stable receipt for an unused test notification path."""
        return DeliveryResult(channel=self.channel_name, message_id="unused")


class NonRepeatableOperation(SourceToolCall):
    """Represent an external mutation that cannot be retried automatically."""

    tool: Literal["non_repeatable_operation"] = "non_repeatable_operation"


class FailingMutationSource:
    """Expose one non-repeatable action that always loses its connection."""

    name = "failing-mutation"
    instructions = "Execute the requested mutation once."
    discovery_tool = "non_repeatable_operation"
    tool_types: tuple[type[SourceToolCall], ...] = (NonRepeatableOperation,)

    def __init__(self) -> None:
        """Start with no mutation attempts."""
        self.attempts = 0

    def parse_tool(self, payload: dict[str, Any]) -> SourceToolCall:
        """Validate the source's only tool payload."""
        return NonRepeatableOperation.model_validate(payload)

    async def execute(self, action: SourceToolCall, context: SourceContext) -> SourceExecution:
        """Fail after recording that one external mutation was attempted."""
        self.attempts += 1
        raise ConnectionError("mutation outcome is unknown")


class MutationAgent:
    """Select the non-repeatable source operation."""

    async def decide(
        self, transcript: list[dict[str, Any]], source_state: dict[str, Any]
    ) -> AgentTurn:
        """Return the single source mutation used by the retry-boundary test."""
        return AgentTurn(
            rationale="Execute the external mutation.",
            action=NonRepeatableOperation(),
        )


async def test_reasoning_node_recovers_from_transient_failure() -> None:
    """Retry a transient model connection failure through LangGraph policy."""
    agent = FlakyFinishAgent()
    source = AdaptyvDataSource(MockFoundryClient.from_fixture())
    graph = build_autonomous_graph(
        agent,
        source,
        NoopSink(),
        retry_max_attempts=3,
        retry_initial_interval_seconds=0,
    )

    result = await AgentRuntime(graph).run(max_cycles=1)

    assert agent.attempts == 2
    assert result.get("cycle_summary") == "Recovered model call."


async def test_repeatable_source_read_recovers_from_transient_failure() -> None:
    """Retry a source tool only after its schema declares repeat execution safe."""
    client = FlakyListFoundryClient.from_fixture()
    graph = build_autonomous_graph(
        DiscoveryThenFinishAgent(),
        AdaptyvDataSource(client),
        NoopSink(),
        retry_max_attempts=3,
        retry_initial_interval_seconds=0,
    )

    result = await AgentRuntime(graph).run(max_cycles=1)

    assert client.list_attempts == 2
    assert [entry["tool"] for entry in result.get("transcript", [])] == ["list_experiments"]


async def test_reasoning_node_times_out() -> None:
    """Cancel a stalled model call instead of leaving the supervisor task hung."""
    graph = build_autonomous_graph(
        HangingAgent(),
        AdaptyvDataSource(MockFoundryClient.from_fixture()),
        NoopSink(),
        external_call_timeout_seconds=0.01,
        retry_max_attempts=1,
    )

    with pytest.raises(NodeTimeoutError, match="Node 'agent' exceeded its run timeout"):
        await AgentRuntime(graph).run(max_cycles=1)


async def test_non_repeatable_source_action_is_not_retried() -> None:
    """Propagate an uncertain mutation failure after exactly one attempt."""
    source = FailingMutationSource()
    graph = build_autonomous_graph(
        MutationAgent(),
        source,
        NoopSink(),
        retry_max_attempts=3,
        retry_initial_interval_seconds=0,
    )

    with pytest.raises(ConnectionError, match="mutation outcome is unknown"):
        await AgentRuntime(graph).run(max_cycles=1)

    assert source.attempts == 1
