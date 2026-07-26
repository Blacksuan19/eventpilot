"""Prove the generic graph accepts tools from an unrelated monitoring source."""

import asyncio
from typing import Any, Literal

from eventpilot.adapters.adaptyv.mock import MockFoundryClient
from eventpilot.adapters.adaptyv.tools import ListExperiments
from eventpilot.core.agent_reasoning import AgentTurn, SendAlert, build_tool_catalog
from eventpilot.core.autonomous import AgentRuntime, build_autonomous_graph
from eventpilot.core.monitoring import SelectObjective
from eventpilot.core.notifications import DeliveryResult, Notification
from eventpilot.sources.adaptyv import AdaptyvDataSource
from eventpilot.sources.base import (
    ResourceSnapshot,
    SourceContext,
    SourceEffect,
    SourceExecution,
    SourceToolCall,
)


class ListWorkflowRuns(SourceToolCall):
    """Discover GitHub Actions workflow runs through a source-defined tool."""

    tool: Literal["list_workflow_runs"] = "list_workflow_runs"
    repository: str


class GetWorkflowRun(SourceToolCall):
    """Inspect one GitHub Actions workflow run through a source-defined tool."""

    parallel_safe = True
    tool: Literal["get_workflow_run"] = "get_workflow_run"
    run_id: str


class GitHubActionsTestSource:
    """Model the interface a `gh` CLI-backed data source would implement."""

    name = "github-actions-test"
    instructions = "Discover workflow runs, inspect failures, and alert operators."
    discovery_tool = "list_workflow_runs"
    tool_types: tuple[type[SourceToolCall], ...] = (ListWorkflowRuns, GetWorkflowRun)

    def parse_tool(self, payload: dict[str, Any]) -> SourceToolCall:
        """Validate one of this source's two unrelated tool schemas."""
        if payload.get("tool") == "list_workflow_runs":
            return ListWorkflowRuns.model_validate(payload)
        if payload.get("tool") == "get_workflow_run":
            return GetWorkflowRun.model_validate(payload)
        raise ValueError(f"Unknown GitHub Actions test tool: {payload.get('tool')}")

    async def execute(self, action: SourceToolCall, context: SourceContext) -> SourceExecution:
        """Return representative JSON that a `gh` subprocess adapter could produce."""
        if isinstance(action, ListWorkflowRuns):
            run = {"id": "4815", "repository": action.repository, "conclusion": "failure"}
            return SourceExecution(
                result={"runs": [run], "total": 1},
                effects=(
                    SourceEffect(
                        "discovery",
                        resources=(
                            ResourceSnapshot(
                                resource_id="4815",
                                status="failure",
                                payload=run,
                            ),
                        ),
                    ),
                ),
            )
        if isinstance(action, GetWorkflowRun):
            return SourceExecution(
                result={"id": action.run_id, "conclusion": "failure", "failed_job": "tests"},
                effects=(
                    SourceEffect(
                        "observation",
                        resource_id=action.run_id,
                        evidence={"status": "failure", "failed_job": "tests"},
                        inspected=True,
                        result_ready=True,
                    ),
                ),
            )
        raise TypeError(type(action).__name__)


class ScriptedGitHubAgent:
    """Select the GitHub-shaped source tools for an end-to-end graph test."""

    def __init__(self) -> None:
        """Create the fixed tool trajectory."""
        self._turns = iter(
            [
                AgentTurn(
                    rationale="Discover failed workflow runs.",
                    action=ListWorkflowRuns(repository="acme/widget"),
                ),
                AgentTurn(
                    rationale="Monitor the discovered workflow run.",
                    action=SelectObjective(
                        kind="monitor",
                        resource_ids=["4815"],
                        summary="Investigate failed workflow runs.",
                    ),
                ),
                AgentTurn(
                    rationale="Inspect the discovered failure.",
                    action=GetWorkflowRun(run_id="4815"),
                ),
                AgentTurn(
                    rationale="Alert after observing the failed job.",
                    action=SendAlert(
                        resource_ids=["4815"],
                        title="GitHub Actions failure",
                        body="Workflow run 4815 failed in the tests job.",
                    ),
                ),
            ]
        )

    async def decide(
        self, transcript: list[dict[str, Any]], source_state: dict[str, Any]
    ) -> AgentTurn:
        """Return the next GitHub-specific tool without graph customization."""
        return next(self._turns)


class ParallelGitHubActionsTestSource(GitHubActionsTestSource):
    """Expose two independent workflow reads and prove they overlap in execution."""

    def __init__(self) -> None:
        """Create a synchronization point and concurrency counters."""
        self._both_started = asyncio.Event()
        self._active = 0
        self.max_active = 0

    async def execute(self, action: SourceToolCall, context: SourceContext) -> SourceExecution:
        """Block each detail read until both LangGraph Send branches have started."""
        if isinstance(action, ListWorkflowRuns):
            runs = [
                {"id": run_id, "repository": action.repository, "conclusion": "failure"}
                for run_id in ("4815", "4816")
            ]
            return SourceExecution(
                result={"runs": runs, "total": len(runs)},
                effects=(
                    SourceEffect(
                        "discovery",
                        resources=tuple(
                            ResourceSnapshot(
                                resource_id=run["id"],
                                status="failure",
                                payload=run,
                            )
                            for run in runs
                        ),
                    ),
                ),
            )
        if isinstance(action, GetWorkflowRun):
            self._active += 1
            self.max_active = max(self.max_active, self._active)
            if self._active == 2:
                self._both_started.set()
            await asyncio.wait_for(self._both_started.wait(), timeout=1)
            execution = await super().execute(action, context)
            self._active -= 1
            return execution
        return await super().execute(action, context)


class ScriptedParallelGitHubAgent:
    """Select two independent workflow reads in one reasoning turn."""

    def __init__(self) -> None:
        """Create a trajectory containing one parallel read batch."""
        self._turns = iter(
            [
                AgentTurn(
                    rationale="Discover failed workflow runs.",
                    action=ListWorkflowRuns(repository="acme/widget"),
                ),
                AgentTurn(
                    rationale="Monitor both discovered workflow runs.",
                    action=SelectObjective(
                        kind="monitor",
                        resource_ids=["4815", "4816"],
                        summary="Investigate both failed workflow runs.",
                    ),
                ),
                AgentTurn(
                    rationale="Inspect both independent failures concurrently.",
                    actions=[GetWorkflowRun(run_id="4815"), GetWorkflowRun(run_id="4816")],
                ),
                AgentTurn(
                    rationale="Report the first observed failure.",
                    action=SendAlert(
                        resource_ids=["4815"],
                        title="GitHub Actions failure 4815",
                        body="Workflow run 4815 failed in the tests job.",
                    ),
                ),
                AgentTurn(
                    rationale="Report the second observed failure.",
                    action=SendAlert(
                        resource_ids=["4816"],
                        title="GitHub Actions failure 4816",
                        body="Workflow run 4816 failed in the tests job.",
                    ),
                ),
            ]
        )

    async def decide(
        self, transcript: list[dict[str, Any]], source_state: dict[str, Any]
    ) -> AgentTurn:
        """Return the next scripted single action or parallel action batch."""
        return next(self._turns)


class RecordingSink:
    """Capture the generic alert emitted from the GitHub-shaped source."""

    channel_name = "recording"

    def __init__(self) -> None:
        """Create an empty notification collection."""
        self.notifications: list[Notification] = []

    async def send(
        self,
        destination: str,
        notification: Notification,
        *,
        idempotency_key: str,
    ) -> DeliveryResult:
        """Record one alert and return a successful delivery receipt."""
        self.notifications.append(notification)
        return DeliveryResult(channel=self.channel_name, message_id="1")


async def test_graph_runs_github_actions_tools_without_platform_changes() -> None:
    """Execute non-Adaptyv tools through the unchanged generic autonomous graph."""
    source = GitHubActionsTestSource()
    sink = RecordingSink()
    graph = build_autonomous_graph(
        ScriptedGitHubAgent(), source, sink, max_wait_seconds=3_600
    )

    result = await AgentRuntime(graph).run(max_invocations=1)

    assert [entry["tool"] for entry in result.get("transcript", [])] == [
        "list_workflow_runs",
        "select_objective",
        "get_workflow_run",
        "send_alert",
    ]
    assert result.get("source_state", {}).get("completed_resource_ids") == ["4815"]
    assert sink.notifications[0].title == "GitHub Actions failure"


async def test_graph_executes_independent_source_actions_in_parallel() -> None:
    """Fan out independent source reads and reduce their effects in selection order."""
    source = ParallelGitHubActionsTestSource()
    sink = RecordingSink()
    graph = build_autonomous_graph(
        ScriptedParallelGitHubAgent(), source, sink, max_wait_seconds=3_600
    )

    result = await AgentRuntime(graph).run(max_invocations=1)

    assert source.max_active == 2
    assert [entry["tool"] for entry in result.get("transcript", [])] == [
        "list_workflow_runs",
        "select_objective",
        "get_workflow_run",
        "get_workflow_run",
        "send_alert",
        "send_alert",
    ]
    assert [entry["call"]["run_id"] for entry in result.get("transcript", [])[2:4]] == [
        "4815",
        "4816",
    ]
    assert result.get("source_state", {}).get("completed_resource_ids") == ["4815", "4816"]
    assert len(sink.notifications) == 2


def test_data_source_publishes_structured_tool_descriptions() -> None:
    """Derive tool documentation from source schemas without handwritten signatures."""
    source = AdaptyvDataSource(MockFoundryClient.from_fixture())

    catalog = build_tool_catalog(source.tool_types)
    list_schema = next(schema for schema in catalog if schema["title"] == "ListExperiments")

    assert ListExperiments in source.tool_types
    assert source.instructions.startswith("Data source: Adaptyv Foundry")
    assert list_schema["description"].startswith("List experiments visible")
    assert list_schema["properties"]["limit"]["description"] == "Maximum records to return."
    assert list_schema["properties"]["limit"]["maximum"] == 100
    detail_schema = next(schema for schema in catalog if schema["title"] == "GetExperiment")
    assert detail_schema["x-parallel-safe"] is True
