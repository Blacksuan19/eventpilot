"""Prove the generic graph accepts tools from an unrelated monitoring source."""

from typing import Any, Literal

from eventpilot.adapters.adaptyv.mock import MockFoundryClient
from eventpilot.adapters.adaptyv.tools import FoundryToolAdapter, ListExperiments
from eventpilot.core.agent_reasoning import AgentTurn, SendAlert, build_tool_catalog
from eventpilot.core.autonomous import AgentRuntime, build_autonomous_graph
from eventpilot.core.monitoring import SelectObjective
from eventpilot.core.notifications import DeliveryResult, Notification
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


class RecordingSink:
    """Capture the generic alert emitted from the GitHub-shaped source."""

    channel_name = "recording"

    def __init__(self) -> None:
        """Create an empty notification collection."""
        self.notifications: list[Notification] = []

    async def send(self, destination: str, notification: Notification) -> DeliveryResult:
        """Record one alert and return a successful delivery receipt."""
        self.notifications.append(notification)
        return DeliveryResult(channel=self.channel_name, message_id="1")


async def test_graph_runs_github_actions_tools_without_platform_changes() -> None:
    """Execute non-Adaptyv tools through the unchanged generic autonomous graph."""
    source = GitHubActionsTestSource()
    sink = RecordingSink()
    graph = build_autonomous_graph(ScriptedGitHubAgent(), source, sink)

    result = await AgentRuntime(graph).run(max_cycles=1)

    assert [entry["tool"] for entry in result.get("transcript", [])] == [
        "list_workflow_runs",
        "select_objective",
        "get_workflow_run",
        "send_alert",
    ]
    assert result.get("source_state", {}).get("completed_resource_ids") == ["4815"]
    assert sink.notifications[0].title == "GitHub Actions failure"


def test_adapter_publishes_structured_tool_descriptions() -> None:
    """Derive tool documentation from adapter schemas without handwritten prompt signatures."""
    adapter = FoundryToolAdapter(MockFoundryClient.from_fixture())

    catalog = build_tool_catalog(adapter.tool_types)
    list_schema = next(schema for schema in catalog if schema["title"] == "ListExperiments")

    assert ListExperiments in adapter.tool_types
    assert adapter.instructions.startswith("Data source: Adaptyv Foundry")
    assert list_schema["description"].startswith("List experiments visible")
    assert list_schema["properties"]["limit"]["description"] == "Maximum records to return."
    assert list_schema["properties"]["limit"]["maximum"] == 100
