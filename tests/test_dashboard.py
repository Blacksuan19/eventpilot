"""Verify the live dashboard event store and HTTP surface."""

from pathlib import Path

from httpx import ASGITransport, AsyncClient

from eventpilot.core.approvals import ApprovalDecision
from eventpilot.core.reporting import AgentActionSelection, AgentDecisionEvent
from eventpilot.dashboard.app import DashboardEventStore, create_dashboard_app


def decision_event() -> AgentDecisionEvent:
    """Build one representative agent decision for dashboard tests."""
    return AgentDecisionEvent(
        data_source="adaptyv-foundry",
        cycle_count=2,
        tool_count=3,
        rationale="Inspect the active experiment before deciding whether to wait.",
        actions=[
            AgentActionSelection(
                tool="get_experiment",
                action_model="GetExperiment",
                arguments={"experiment_id": "exp-protein-001"},
            )
        ],
        parallel=False,
        available_tools=["get_experiment", "wait"],
        source_state={"phase": "monitoring", "monitoring": {"exp-protein-001": {}}},
    )


def test_store_retains_complete_typed_events() -> None:
    """Keep full agent state available to browser exploration."""
    store = DashboardEventStore(max_events=2)

    store.emit(decision_event())

    [event] = store.snapshot()
    assert event["event"] == "agent_decision"
    assert event["actions"][0]["arguments"] == {"experiment_id": "exp-protein-001"}
    assert event["source_state"]["monitoring"] == {"exp-protein-001": {}}


def test_store_restores_durable_event_history(tmp_path: Path) -> None:
    """Restore delivered runtime events after a dashboard process restart."""
    event_log = tmp_path / "agent.events.jsonl"
    first_store = DashboardEventStore(path=event_log)
    first_store.emit(decision_event())

    restored_store = DashboardEventStore(path=event_log)

    assert restored_store.snapshot() == first_store.snapshot()

    restored_store.clear()

    assert restored_store.snapshot() == []
    assert not event_log.exists()


async def test_dashboard_serves_page_health_and_history() -> None:
    """Expose a ready presentation page and its initial event history."""
    store = DashboardEventStore()
    store.emit(decision_event())
    reset_calls = 0
    approval_calls: list[tuple[str, ApprovalDecision]] = []

    async def reset_agent() -> None:
        """Record one dashboard reset request."""
        nonlocal reset_calls
        reset_calls += 1

    async def resolve_approval(approval_id: str, decision: ApprovalDecision) -> bool:
        """Record one operator decision from the dashboard."""
        approval_calls.append((approval_id, decision))
        return approval_id == "approval-1"

    async with AsyncClient(
        transport=ASGITransport(
            app=create_dashboard_app(
                store,
                reset_agent=reset_agent,
                resolve_approval=resolve_approval,
            )
        ),
        base_url="http://test",
    ) as client:
        page = await client.get("/")
        health = await client.get("/api/health")
        history = await client.get("/api/events")
        reset = await client.post("/api/reset")
        approval = await client.post("/api/approvals/approval-1", json={"decision": "approved"})
        missing_approval = await client.post(
            "/api/approvals/missing", json={"decision": "rejected"}
        )

    assert page.status_code == 200
    assert "EventPilot" in page.text
    assert "Current agent activity" in page.text
    assert "Reset demo" in page.text
    assert ".activity { height:360px" in page.text
    assert "#arguments { flex:1; min-height:52px; overflow:auto" in page.text
    assert "e.event==='cycle_finished'?'Cycle finished'" in page.text
    assert "e.event==='tool_result'&&e.tool==='send_alert'&&e.result?.message_id" in page.text
    assert health.json() == {"status": "ok"}
    assert history.json()["events"][0]["actions"][0]["tool"] == "get_experiment"
    assert reset.json() == {"status": "reset"}
    assert reset_calls == 1
    assert approval.json() == {"status": "approved"}
    assert missing_approval.status_code == 404
    assert approval_calls == [
        ("approval-1", ApprovalDecision.APPROVED),
        ("missing", ApprovalDecision.REJECTED),
    ]
