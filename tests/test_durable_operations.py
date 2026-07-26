"""Verify external side effects survive LangGraph node replay without duplication."""

from typing import Any, Literal, TypedDict

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from eventpilot.core.autonomous.tasks import DurableOperations
from eventpilot.core.notifications import DeliveryResult, Notification
from eventpilot.notifications.console import ConsoleNotificationSink
from eventpilot.sources.base import SourceContext, SourceExecution, SourceToolCall


class Mutation(SourceToolCall):
    """Represent one remote mutation used to detect duplicate execution."""

    tool: Literal["mutate"] = "mutate"


class CountingSource:
    """Count executions and expose the operation identity received from the graph."""

    name = "counting-source"
    instructions = "Execute the mutation."
    discovery_tool = "mutate"
    tool_types: tuple[type[SourceToolCall], ...] = (Mutation,)

    def __init__(self) -> None:
        """Start with no external mutation attempts."""
        self.attempts = 0
        self.operation_ids: list[str | None] = []

    def parse_tool(self, payload: dict[str, Any]) -> SourceToolCall:
        """Validate the source's mutation payload."""
        return Mutation.model_validate(payload)

    async def execute(self, action: SourceToolCall, context: SourceContext) -> SourceExecution:
        """Record one simulated remote mutation and its idempotency identity."""
        self.attempts += 1
        self.operation_ids.append(context.operation_id)
        return SourceExecution(result={"status": "mutated"})


class CountingSink:
    """Count notification provider calls without performing its own deduplication."""

    channel_name = "counting"

    def __init__(self) -> None:
        """Start with no delivery attempts."""
        self.attempts = 0
        self.idempotency_keys: list[str] = []

    async def send(
        self,
        destination: str,
        notification: Notification,
        *,
        idempotency_key: str,
    ) -> DeliveryResult:
        """Record one provider call and return a synthetic receipt."""
        self.attempts += 1
        self.idempotency_keys.append(idempotency_key)
        return DeliveryResult(channel=self.channel_name, message_id="message-1")


class ReplayState(TypedDict, total=False):
    """Hold the durable result of the replay test node."""

    result: str


class SideEffectReplayNode:
    """Execute durable operations and optionally fail after both complete."""

    def __init__(self, operations: DurableOperations, *, fail_after_tasks: bool) -> None:
        """Bind task-backed operations and the simulated failure behavior."""
        self._operations = operations
        self._fail_after_tasks = fail_after_tasks

    async def __call__(self, state: ReplayState, runtime: Runtime[Any]) -> ReplayState:
        """Run source and notification tasks in a stable deterministic order."""
        if runtime.execution_info is None:
            raise RuntimeError("Expected graph execution metadata")
        operation_id = runtime.execution_info.task_id
        completed = await self._operations.execute_source(
            Mutation(),
            SourceContext(state={}, transcript=[], clock=lambda: 0.0),
            operation_id=f"{operation_id}:source",
        )
        receipt = await self._operations.deliver_notification(
            "operator",
            Notification(title="Mutation complete", body="The mutation completed."),
            operation_id=f"{operation_id}:alert",
        )
        if self._fail_after_tasks:
            raise RuntimeError("fail after completed side effects")
        return {"result": f"{completed.execution.result['status']}:{receipt.message_id}"}


def build_replay_graph(node: SideEffectReplayNode, checkpointer: InMemorySaver) -> Any:
    """Compile the replay fixture with a stable node and task ordering."""
    builder = StateGraph(ReplayState)
    builder.add_node("side_effects", node)
    builder.add_edge(START, "side_effects")
    builder.add_edge("side_effects", END)
    return builder.compile(checkpointer=checkpointer)


async def test_completed_side_effect_tasks_are_restored_during_node_replay() -> None:
    """Rebuild and resume a failed graph without repeating its completed side effects."""
    first_source = CountingSource()
    first_sink = CountingSink()
    checkpointer = InMemorySaver()
    graph = build_replay_graph(
        SideEffectReplayNode(
            DurableOperations(first_source, first_sink, lambda: 0.0),
            fail_after_tasks=True,
        ),
        checkpointer,
    )
    config: RunnableConfig = {"configurable": {"thread_id": "side-effect-replay"}}

    with pytest.raises(RuntimeError, match="fail after completed side effects"):
        await graph.ainvoke({}, config)

    restarted_source = CountingSource()
    restarted_sink = CountingSink()
    restarted_graph = build_replay_graph(
        SideEffectReplayNode(
            DurableOperations(restarted_source, restarted_sink, lambda: 0.0),
            fail_after_tasks=False,
        ),
        checkpointer,
    )
    result = await restarted_graph.ainvoke(None, config)

    assert result["result"] == "mutated:message-1"
    assert first_source.attempts == 1
    assert first_sink.attempts == 1
    assert restarted_source.attempts == 0
    assert restarted_sink.attempts == 0
    assert first_source.operation_ids[0] is not None
    assert first_source.operation_ids[0].endswith(":source")
    assert first_sink.idempotency_keys[0].endswith(":alert")


async def test_console_sink_deduplicates_an_in_process_provider_retry(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Return the first receipt when the provider receives the same operation twice."""
    sink = ConsoleNotificationSink()
    notification = Notification(title="Ready", body="Results are ready.")

    first = await sink.send("operator", notification, idempotency_key="operation-1")
    second = await sink.send("operator", notification, idempotency_key="operation-1")

    assert second == first
    assert capsys.readouterr().out.count("Results are ready.") == 1
