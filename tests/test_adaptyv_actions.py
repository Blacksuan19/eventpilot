"""Exercise source mutations and human approval against the Foundry mock."""

import asyncio
from contextlib import suppress
from pathlib import Path
from typing import Any

import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from eventpilot.adapters.adaptyv.mock import MockFoundryClient, load_scenarios
from eventpilot.adapters.adaptyv.models import ModifyExperimentRequest
from eventpilot.adapters.adaptyv.tools import (
    AcceptExperimentQuote,
    GetExperiment,
    GetExperimentQuote,
    ListExperiments,
    UpdateExperiment,
)
from eventpilot.core.agent_reasoning import AgentTurn, FinishCycle
from eventpilot.core.approvals import ApprovalDecision
from eventpilot.core.autonomous import AgentRuntime, build_autonomous_graph
from eventpilot.core.monitoring import (
    SelectObjective,
    apply_execution,
    available_source_tools,
    initial_state,
    select_objective,
)
from eventpilot.core.notifications import DeliveryResult, Notification
from eventpilot.core.reporting import AgentEvent, ApprovalRequestedEvent
from eventpilot.sources.adaptyv import AdaptyvDataSource, AdaptyvSourcePolicy
from eventpilot.sources.base import SourceContext


class StaticClock:
    """Provide a mutable monotonic clock for action-gated fixtures."""

    def __init__(self) -> None:
        """Start fixture time at zero."""
        self.now = 0.0

    def __call__(self) -> float:
        """Return the current fixture time."""
        return self.now


class RecordingSink:
    """Capture approval requests delivered by the graph."""

    channel_name = "recording"

    def __init__(self) -> None:
        """Create an empty notification list."""
        self.notifications: list[Notification] = []

    async def send(self, destination: str, notification: Notification) -> DeliveryResult:
        """Record a notification and return a synthetic provider receipt."""
        self.notifications.append(notification)
        return DeliveryResult(channel=self.channel_name, message_id=str(len(self.notifications)))


class RecordingReporter:
    """Capture approval identifiers published by the graph."""

    def __init__(self) -> None:
        """Create an empty event list."""
        self.events: list[AgentEvent] = []

    def emit(self, event: AgentEvent) -> None:
        """Append one runtime event."""
        self.events.append(event)


class QuoteActionAgent:
    """Drive the real Adaptyv source through quote inspection and acceptance."""

    def __init__(self) -> None:
        """Build a trajectory scoped to the complete discovered portfolio."""
        experiment_ids = [scenario.experiment.id for scenario in load_scenarios()]
        self._turns = iter(
            [
                AgentTurn(rationale="Discover work.", action=ListExperiments()),
                AgentTurn(
                    rationale="Select every active experiment.",
                    action=SelectObjective(
                        kind="monitor",
                        resource_ids=experiment_ids,
                        summary="Monitor the full portfolio.",
                    ),
                ),
                AgentTurn(
                    rationale="Inspect the quoted experiment.",
                    action=GetExperiment(experiment_id="experiment-il6"),
                ),
                AgentTurn(
                    rationale="Read its price before requesting approval.",
                    action=GetExperimentQuote(experiment_id="experiment-il6"),
                ),
                AgentTurn(
                    rationale="Request approval and accept the quote.",
                    action=AcceptExperimentQuote(experiment_id="experiment-il6"),
                ),
                AgentTurn(
                    rationale="The bounded approval scenario is complete.",
                    action=FinishCycle(summary="Quote decision recorded."),
                ),
            ]
        )

    async def decide(
        self, transcript: list[dict[str, Any]], source_state: dict[str, Any]
    ) -> AgentTurn:
        """Return the next typed action in the approval trajectory."""
        return next(self._turns)


class FinishAfterApprovalAgent:
    """Finish a restored cycle after its suspended tool has executed."""

    async def decide(
        self, transcript: list[dict[str, Any]], source_state: dict[str, Any]
    ) -> AgentTurn:
        """Close the bounded cycle after LangGraph resumes the approved action."""
        return AgentTurn(
            rationale="The restored approval scenario is complete.",
            action=FinishCycle(summary="Restored quote approval completed."),
        )


async def test_mock_mutations_gate_draft_and_quote_lifecycles() -> None:
    """Keep actionable fixtures paused until their documented mutations occur."""
    clock = StaticClock()
    client = MockFoundryClient.from_fixture(clock=clock)

    clock.now = 1_000_000
    assert (await client.get_experiment("experiment-tp53")).status == "Draft"
    assert (await client.get_experiment("experiment-il6")).status == "QuoteSent"

    updated = await client.update_experiment(
        "experiment-tp53", ModifyExperimentRequest(n_replicates=3)
    )
    submitted = await client.submit_experiment("experiment-tp53")
    quote = await client.get_experiment_quote("experiment-il6")
    accepted = await client.accept_experiment_quote("experiment-il6")

    assert updated.experiment_spec["n_replicates"] == 3
    assert submitted.previous_status == "Draft"
    assert submitted.status == "WaitingForConfirmation"
    assert quote.amount_total == 125_000
    assert accepted.status == "accepted"
    assert (await client.get_experiment("experiment-il6")).status == "WaitingForMaterials"


async def test_submission_policy_is_structured_and_controls_tool_availability() -> None:
    """Expose source policy as evidence and enforce it through generic graph tooling."""
    clock = StaticClock()
    clock.now = 1_000_000
    source = AdaptyvDataSource(
        MockFoundryClient.from_fixture(clock=clock),
        policy=AdaptyvSourcePolicy(minimum_replicates=4),
    )
    state = initial_state()
    context = SourceContext(state=state, transcript=[], clock=clock)

    discovery = await source.execute(ListExperiments(), context)
    result, state = apply_execution(discovery, state, observed_at=clock())
    _, state = select_objective(
        SelectObjective(
            kind="monitor",
            resource_ids=[item["id"] for item in result["items"]],
            summary="Monitor the Foundry portfolio.",
        ),
        state,
    )
    detail = await source.execute(
        GetExperiment(experiment_id="experiment-tp53"),
        SourceContext(state=state, transcript=[], clock=clock),
    )
    _, state = apply_execution(detail, state, observed_at=clock())

    requirement = state["evidence"]["experiment-tp53"]["requirements"]["submit_experiment"]
    assert requirement == {
        "minimum_replicates": 4,
        "actual_replicates": 1,
        "satisfied": False,
    }
    assert "update_experiment" in available_source_tools(source, state)
    assert "submit_experiment" not in available_source_tools(source, state)

    update = await source.execute(
        UpdateExperiment(
            experiment_id="experiment-tp53",
            changes=ModifyExperimentRequest(n_replicates=4),
        ),
        SourceContext(state=state, transcript=[], clock=clock),
    )
    _, state = apply_execution(update, state, observed_at=clock())

    requirement = state["evidence"]["experiment-tp53"]["requirements"]["submit_experiment"]
    assert requirement["satisfied"] is True
    assert "submit_experiment" in available_source_tools(source, state)


@pytest.mark.parametrize(
    ("decision", "expected_status"),
    [
        (ApprovalDecision.APPROVED, "accepted"),
        (ApprovalDecision.REJECTED, "rejected"),
    ],
)
async def test_quote_action_waits_for_operator_decision(
    decision: ApprovalDecision, expected_status: str
) -> None:
    """Deliver an approval request before executing or rejecting quote acceptance."""
    client = MockFoundryClient.from_fixture()
    source = AdaptyvDataSource(client)
    sink = RecordingSink()
    reporter = RecordingReporter()
    graph = build_autonomous_graph(
        QuoteActionAgent(),
        source,
        sink,
        reporter=reporter,
    )
    runtime = AgentRuntime(graph)

    run = asyncio.create_task(runtime.run(max_cycles=1))
    for _ in range(100):
        requested = next(
            (event for event in reporter.events if isinstance(event, ApprovalRequestedEvent)),
            None,
        )
        if requested is not None:
            break
        if run.done():
            await run
        await asyncio.sleep(0.01)
    else:
        pytest.fail("The graph did not request quote approval")

    assert not run.done()
    assert sink.notifications[-1].title == "Approve quote for experiment-il6"
    assert "USD 1,250.00" in sink.notifications[-1].body
    assert await runtime.resolve_approval(requested.approval_id, decision)

    result = await asyncio.wait_for(run, timeout=1)
    acceptance = next(
        entry
        for entry in result.get("transcript", [])
        if entry["tool"] == "accept_experiment_quote"
    )
    assert acceptance["result"]["status"] == expected_status
    if decision is ApprovalDecision.APPROVED:
        assert (await client.get_experiment("experiment-il6")).status == "WaitingForMaterials"
    else:
        assert (await client.get_experiment("experiment-il6")).status == "QuoteSent"


async def test_quote_interrupt_resumes_after_runtime_restart(tmp_path: Path) -> None:
    """Persist a pending approval and resume it through a newly compiled graph."""
    database = tmp_path / "checkpoints.sqlite"
    first_sink = RecordingSink()
    first_reporter = RecordingReporter()
    config = {"configurable": {"thread_id": AgentRuntime.thread_id}}

    async with AsyncSqliteSaver.from_conn_string(str(database)) as checkpointer:
        first_graph = build_autonomous_graph(
            QuoteActionAgent(),
            AdaptyvDataSource(MockFoundryClient.from_fixture()),
            first_sink,
            reporter=first_reporter,
            checkpointer=checkpointer,
        )
        first_run = asyncio.create_task(AgentRuntime(first_graph).run(max_cycles=1))
        for _ in range(100):
            snapshot = await first_graph.aget_state(config)
            if any(task.interrupts for task in snapshot.tasks):
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("The first graph did not persist its approval interrupt")
        pending = snapshot.values["pending_approval"]
        first_run.cancel()
        with suppress(asyncio.CancelledError):
            await first_run

    second_sink = RecordingSink()
    second_client = MockFoundryClient.from_fixture()
    async with AsyncSqliteSaver.from_conn_string(str(database)) as checkpointer:
        restored_graph = build_autonomous_graph(
            FinishAfterApprovalAgent(),
            AdaptyvDataSource(second_client),
            second_sink,
            checkpointer=checkpointer,
        )
        restored_runtime = AgentRuntime(restored_graph)
        assert await restored_runtime.resolve_approval(
            str(pending["id"]), ApprovalDecision.APPROVED
        )

        result = await restored_runtime.run(max_cycles=1)

    acceptance = next(
        entry
        for entry in result.get("transcript", [])
        if entry["tool"] == "accept_experiment_quote"
    )
    assert acceptance["result"]["status"] == "accepted"
    assert len(first_sink.notifications) == 1
    assert second_sink.notifications == []
