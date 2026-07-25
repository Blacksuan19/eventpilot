"""Exercise autonomous trajectories, persistence, and Foundry HTTP integration."""

from pathlib import Path

import httpx
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from eventpilot.adapters.adaptyv import FoundryHttpClient
from eventpilot.core.agent_reasoning import DemoAutonomousReasoningEngine
from eventpilot.core.autonomous import AgentRuntime, build_autonomous_graph
from eventpilot.core.notifications import DeliveryResult, Notification
from eventpilot.simulator import ProgressingFoundryClient, tool_names


class RecordingSink:
    """Capture autonomous update actions without external side effects."""

    channel_name = "recording"

    def __init__(self) -> None:
        """Create an empty update history."""
        self.notifications: list[Notification] = []

    async def send(self, destination: str, notification: Notification) -> DeliveryResult:
        """Store one delivered update and return its synthetic receipt."""
        self.notifications.append(notification)
        return DeliveryResult(channel=self.channel_name, message_id=str(len(self.notifications)))


async def no_sleep(delay: float) -> None:
    """Complete agent-selected waits immediately during trajectory tests."""


async def test_agent_discovers_polls_results_and_finishes_cycle() -> None:
    """Prove the LLM control loop chooses the complete autonomous tool trajectory."""
    foundry = ProgressingFoundryClient(["InQueue", "InProduction", "Done"])
    sink = RecordingSink()
    graph = build_autonomous_graph(DemoAutonomousReasoningEngine(), foundry, sink, sleep=no_sleep)

    result = await AgentRuntime(graph).run(max_cycles=1)

    transcript = result.get("transcript")
    assert transcript is not None
    assert tool_names(transcript) == [
        "list_experiments",
        "get_experiment",
        "wait",
        "get_experiment",
        "wait",
        "get_experiment",
        "list_experiment_results",
        "send_update",
    ]
    assert result.get("outcome") == "cycle_finished"
    assert result.get("cycle_count") == 1
    assert len(sink.notifications) == 1
    assert "results are ready" in sink.notifications[0].title


async def test_sqlite_checkpoint_survives_fresh_cycles(tmp_path: Path) -> None:
    """Prove fresh invocations retain durable counters while clearing working context."""
    sink = RecordingSink()
    database = tmp_path / "checkpoints.sqlite"
    async with AsyncSqliteSaver.from_conn_string(str(database)) as checkpointer:
        graph = build_autonomous_graph(
            DemoAutonomousReasoningEngine(),
            ProgressingFoundryClient(["InQueue", "Done"]),
            sink,
            checkpointer=checkpointer,
            sleep=no_sleep,
        )
        first = await AgentRuntime(graph).run(max_cycles=1)
        restarted_graph = build_autonomous_graph(
            DemoAutonomousReasoningEngine(),
            ProgressingFoundryClient(["InQueue", "Done"]),
            sink,
            checkpointer=checkpointer,
            sleep=no_sleep,
        )
        second = await AgentRuntime(restarted_graph).run(max_cycles=1)

    assert first.get("cycle_count") == 1
    assert second.get("cycle_count") == 2
    assert len(second.get("transcript", [])) == 6


async def test_http_client_calls_documented_foundry_paths() -> None:
    """Validate authentication, paths, and response parsing against a mock transport."""

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer test-token"
        if request.url.path == "/api/v1/experiments":
            experiment = ProgressingFoundryClient(["InQueue"])
            page = await experiment.list_experiments()
            return httpx.Response(200, json=page.model_dump(mode="json"))
        return httpx.Response(404, json={"error": "not found", "request_id": "req-test"})

    client = FoundryHttpClient(
        "https://example.test/api/v1",
        "test-token",
        transport=httpx.MockTransport(handler),
    )
    async with client:
        page = await client.list_experiments()

    assert page.count == 1
    assert page.items[0].status == "InQueue"
