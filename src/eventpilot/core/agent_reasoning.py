"""Define validated agent tool choices and LLM reasoning engines."""

import json
from typing import Annotated, Any, Literal, Protocol, cast

import instructor
from instructor import AsyncInstructor
from pydantic import BaseModel, ConfigDict, Field

from eventpilot.adapters.adaptyv import ExperimentStatus
from eventpilot.core.notifications import NotificationPriority
from eventpilot.prompts.loader import load_prompt


class ListExperiments(BaseModel):
    """Discover experiments accessible to the Foundry organization."""

    tool: Literal["list_experiments"] = "list_experiments"
    limit: int = Field(default=50, ge=1, le=100)
    offset: int = Field(default=0, ge=0)


class GetExperiment(BaseModel):
    """Retrieve the current detailed state of one experiment."""

    tool: Literal["get_experiment"] = "get_experiment"
    experiment_id: str = Field(min_length=1)


class ListExperimentUpdates(BaseModel):
    """Retrieve chronological progress and error updates for one experiment."""

    tool: Literal["list_experiment_updates"] = "list_experiment_updates"
    experiment_id: str = Field(min_length=1)


class ListExperimentResults(BaseModel):
    """Retrieve analysis results currently available for one experiment."""

    tool: Literal["list_experiment_results"] = "list_experiment_results"
    experiment_id: str = Field(min_length=1)


class SendUpdate(BaseModel):
    """Send an operator update through the configured trusted destination."""

    tool: Literal["send_update"] = "send_update"
    title: str = Field(min_length=1)
    body: str = Field(min_length=1)
    priority: NotificationPriority = NotificationPriority.NORMAL


class Wait(BaseModel):
    """Pause the current objective before the agent chooses another tool."""

    tool: Literal["wait"] = "wait"
    seconds: int = Field(ge=1, le=86_400)
    reason: str = Field(min_length=1)


class FinishCycle(BaseModel):
    """Complete the objective and return control to the fresh-cycle runtime."""

    tool: Literal["finish_cycle"] = "finish_cycle"
    summary: str = Field(min_length=1)


AgentToolCall = Annotated[
    ListExperiments
    | GetExperiment
    | ListExperimentUpdates
    | ListExperimentResults
    | SendUpdate
    | Wait
    | FinishCycle,
    Field(discriminator="tool"),
]


class AgentTurn(BaseModel):
    """Represent one validated tool choice made by the autonomous LLM."""

    model_config = ConfigDict(frozen=True)

    rationale: str = Field(min_length=1)
    action: AgentToolCall


class AutonomousReasoningEngine(Protocol):
    """Choose the next tool from the autonomous agent's accumulated context."""

    async def decide(self, transcript: list[dict[str, Any]]) -> AgentTurn:
        """Return one validated tool call without executing it."""
        ...


class InstructorAutonomousReasoningEngine:
    """Use Instructor to make provider-neutral, schema-validated agent tool choices."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        max_tool_calls_per_cycle: int = 32,
    ) -> None:
        """Create the asynchronous Instructor client used for every agent turn."""
        options: dict[str, Any] = {}
        if api_key:
            options["api_key"] = api_key
        if api_base:
            options["base_url"] = api_base
        self._client = cast(
            AsyncInstructor,
            instructor.from_provider(model, async_client=True, **options),
        )
        self._max_tool_calls_per_cycle = max_tool_calls_per_cycle

    async def decide(self, transcript: list[dict[str, Any]]) -> AgentTurn:
        """Ask the LLM to inspect tool results and select exactly one next tool."""
        if len(transcript) >= self._max_tool_calls_per_cycle:
            return AgentTurn(
                rationale="The cycle reached its tool budget and must yield to a fresh cycle.",
                action=FinishCycle(
                    summary="Cycle tool budget reached; resume from fresh evidence."
                ),
            )
        return await self._client.create(
            response_model=AgentTurn,
            messages=[
                {"role": "system", "content": load_prompt("autonomous_agent.txt")},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "remaining_tool_calls": self._max_tool_calls_per_cycle
                            - len(transcript),
                            "tool_transcript": transcript,
                        }
                    ),
                },
            ],
            max_retries=2,
        )


class DemoAutonomousReasoningEngine:
    """Exercise the complete tool loop deterministically without LLM credentials."""

    async def decide(self, transcript: list[dict[str, Any]]) -> AgentTurn:
        """Choose representative discovery, inspection, wait, update, and finish calls."""
        if not transcript:
            return _list_experiments_turn()

        latest = transcript[-1]
        tool = latest["tool"]
        if tool == "list_experiments":
            items = latest["result"]["items"]
            active = [
                item
                for item in items
                if item["status"] not in {ExperimentStatus.DONE, ExperimentStatus.CANCELED}
            ]
            if not active:
                return AgentTurn(
                    rationale="No active experiment currently requires attention.",
                    action=Wait(seconds=60, reason="Back off before discovering work again."),
                )
            experiment_id = active[0]["id"]
            return AgentTurn(
                rationale="Inspect the newest active experiment before deciding what to do.",
                action=GetExperiment(experiment_id=experiment_id),
            )

        if tool == "wait":
            experiment_id = _last_active_experiment_id(transcript)
            if experiment_id:
                return AgentTurn(
                    rationale=(
                        "The selected polling interval elapsed; refresh the active experiment."
                    ),
                    action=GetExperiment(experiment_id=experiment_id),
                )
            return _list_experiments_turn()

        if tool == "get_experiment":
            experiment = latest["result"]
            experiment_id = experiment["id"]
            if experiment["results_status"] in {"Partial", "All"}:
                return AgentTurn(
                    rationale=(
                        "Results are available and should be inspected before updating anyone."
                    ),
                    action=ListExperimentResults(experiment_id=experiment_id),
                )
            if experiment["status"] == ExperimentStatus.DONE:
                return AgentTurn(
                    rationale=(
                        "The experiment is done but results are not yet exposed; inspect updates."
                    ),
                    action=ListExperimentUpdates(experiment_id=experiment_id),
                )
            return AgentTurn(
                rationale=f"The experiment remains active in {experiment['status']}.",
                action=Wait(seconds=1, reason="Poll this active experiment again."),
            )

        if tool == "list_experiment_updates":
            call = latest["call"]
            return AgentTurn(
                rationale=(
                    "The terminal experiment has no results yet, so it should be checked later."
                ),
                action=Wait(
                    seconds=5,
                    reason=f"Wait for results on experiment {call['experiment_id']}.",
                ),
            )

        if tool == "list_experiment_results":
            result_page = latest["result"]
            call = latest["call"]
            return AgentTurn(
                rationale="The result payload confirms that operator-ready data is available.",
                action=SendUpdate(
                    title=f"Experiment {call['experiment_id']} results are ready",
                    body=f"Foundry returned {result_page['count']} available result(s).",
                ),
            )

        if tool == "send_update":
            return AgentTurn(
                rationale="The operator update was delivered and this objective is complete.",
                action=FinishCycle(summary="Inspected results and updated the operator."),
            )

        return AgentTurn(
            rationale="Begin a fresh discovery pass.",
            action=ListExperiments(),
        )


def _list_experiments_turn() -> AgentTurn:
    """Return the initial discovery action used by the offline agent."""
    return AgentTurn(
        rationale="Discover current experiments before selecting an objective.",
        action=ListExperiments(),
    )


def _last_active_experiment_id(transcript: list[dict[str, Any]]) -> str | None:
    """Recover the most recent nonterminal experiment selected by the offline agent."""
    for entry in reversed(transcript):
        if entry["tool"] != "get_experiment":
            continue
        experiment = entry["result"]
        if experiment["status"] not in {ExperimentStatus.DONE, ExperimentStatus.CANCELED}:
            return str(experiment["id"])
    return None
