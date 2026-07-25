"""Define generic agent tools and provider-neutral structured reasoning."""

import json
from functools import reduce
from operator import or_
from typing import Annotated, Any, Literal, Protocol, cast

import instructor
from instructor import AsyncInstructor
from pydantic import BaseModel, ConfigDict, Field, SerializeAsAny, create_model

from eventpilot.core.notifications import NotificationPriority
from eventpilot.prompts.loader import load_prompt
from eventpilot.sources.base import DataSource, SourceToolCall


class SendAlert(SourceToolCall):
    """Send an operator alert through the configured notification sink."""

    tool: Literal["send_alert"] = "send_alert"
    resource_ids: list[str] = Field(
        min_length=1, description="Platform resource identifiers discussed in the alert."
    )
    title: str = Field(min_length=1, description="Concise operator-facing alert title.")
    body: str = Field(min_length=1, description="Evidence-based operator-facing alert body.")
    priority: NotificationPriority = Field(
        default=NotificationPriority.NORMAL, description="Delivery urgency."
    )


class Wait(SourceToolCall):
    """Pause current work before allowing the agent to select another tool."""

    tool: Literal["wait"] = "wait"
    seconds: int = Field(ge=1, le=86_400, description="Requested polling delay in seconds.")
    reason: str = Field(min_length=1, description="Why current work requires a pause.")


class FinishCycle(SourceToolCall):
    """Complete current work and return control to the fresh-cycle runtime."""

    tool: Literal["finish_cycle"] = "finish_cycle"
    summary: str = Field(min_length=1, description="Completed work or yield reason.")


CORE_TOOL_TYPES: tuple[type[SourceToolCall], ...] = (SendAlert, Wait, FinishCycle)


class AgentTurn(BaseModel):
    """Represent one validated tool choice made by the autonomous agent."""

    model_config = ConfigDict(frozen=True)

    rationale: str = Field(min_length=1)
    action: SerializeAsAny[SourceToolCall]


class AutonomousReasoningEngine(Protocol):
    """Choose the next registered tool from accumulated autonomous context."""

    async def decide(
        self, transcript: list[dict[str, Any]], source_state: dict[str, Any]
    ) -> AgentTurn:
        """Return exactly one typed tool call without executing it."""
        ...


class InstructorAutonomousReasoningEngine:
    """Use Instructor to select from core tools and a data source's typed tools."""

    def __init__(
        self,
        model: str,
        source: DataSource,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        max_tool_calls_per_cycle: int = 32,
    ) -> None:
        """Create a structured client and dynamic response schema for one source plugin."""
        options: dict[str, Any] = {}
        if api_key:
            options["api_key"] = api_key
        if api_base:
            options["base_url"] = api_base
        self._client = cast(
            AsyncInstructor,
            instructor.from_provider(model, async_client=True, **options),
        )
        self._source = source
        self._max_tool_calls_per_cycle = max_tool_calls_per_cycle

    async def decide(
        self, transcript: list[dict[str, Any]], source_state: dict[str, Any]
    ) -> AgentTurn:
        """Ask the LLM to inspect source state and select one registered tool."""
        finish_rejection = self._source.validate_finish(
            source_state,
            tool_count=len(transcript),
            max_tool_calls=self._max_tool_calls_per_cycle,
        )
        if len(transcript) >= self._max_tool_calls_per_cycle and finish_rejection is None:
            return AgentTurn(
                rationale="The cycle reached its tool budget and must yield to a fresh cycle.",
                action=FinishCycle(
                    summary="Cycle tool budget reached; resume from fresh source evidence."
                ),
            )
        available_types = available_tool_types(
            self._source,
            source_state,
            tool_count=len(transcript),
            max_tool_calls=self._max_tool_calls_per_cycle,
        )
        action_union = reduce(or_, available_types)
        action_type = Annotated[action_union, Field(discriminator="tool")]
        response_model = create_model(
            f"{self._source.name.title().replace('-', '')}AvailableAgentTurn",
            rationale=(str, Field(min_length=1)),
            action=(action_type, ...),
        )
        response: Any = await self._client.create(
            response_model=response_model,
            messages=[
                {
                    "role": "system",
                    "content": "\n\n".join(
                        [load_prompt("autonomous_agent.txt"), self._source.instructions]
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "data_source": self._source.name,
                            "currently_available_tools": sorted(
                                tool_type.model_fields["tool"].default
                                for tool_type in available_types
                            ),
                            "tool_catalog": build_tool_catalog(available_types),
                            "remaining_tool_calls": self._max_tool_calls_per_cycle
                            - len(transcript),
                            "source_state": source_state,
                            "tool_transcript": transcript,
                        }
                    ),
                },
            ],
            max_retries=2,
        )
        return AgentTurn(rationale=response.rationale, action=response.action)


def available_tool_types(
    source: DataSource,
    source_state: dict[str, Any],
    *,
    tool_count: int,
    max_tool_calls: int,
) -> tuple[type[SourceToolCall], ...]:
    """Return only tool schemas permitted by current deterministic graph policy."""
    source_names = source.available_tools(source_state)
    tools = tuple(
        tool_type
        for tool_type in source.tool_types
        if tool_type.model_fields["tool"].default in source_names
    )
    core: tuple[type[SourceToolCall], ...] = (SendAlert,)
    if source.validate_wait(source_state) is None:
        core += (Wait,)
    if (
        source.validate_finish(
            source_state,
            tool_count=tool_count,
            max_tool_calls=max_tool_calls,
        )
        is None
    ):
        core += (FinishCycle,)
    return (*tools, *core)


def build_tool_catalog(
    tool_types: tuple[type[SourceToolCall], ...],
) -> list[dict[str, Any]]:
    """Derive the agent-visible tool catalog directly from Pydantic schemas."""
    return [tool_type.model_json_schema() for tool_type in tool_types]


def parse_core_tool(payload: dict[str, Any]) -> SourceToolCall | None:
    """Validate a persisted core tool payload, returning none for source tools."""
    tool_types = {
        tool_type.model_fields["tool"].default: tool_type for tool_type in CORE_TOOL_TYPES
    }
    tool_type = tool_types.get(payload.get("tool"))
    return tool_type.model_validate(payload) if tool_type else None
