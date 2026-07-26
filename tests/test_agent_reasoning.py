"""Verify provider-neutral structured reasoning can select action batches."""

from typing import Any

import instructor

from eventpilot.adapters.adaptyv.mock import MockFoundryClient
from eventpilot.core.agent_reasoning import (
    InstructorAutonomousReasoningEngine,
    available_tool_types,
)
from eventpilot.core.monitoring import initial_state
from eventpilot.sources.adaptyv import AdaptyvDataSource


class BatchResponseClient:
    """Return two source actions through the dynamic Instructor response model."""

    response_schema: dict[str, Any] | None = None

    async def create(self, *, response_model: type[Any], **kwargs: Any) -> Any:
        """Validate a representative parallel action response."""
        self.response_schema = response_model.model_json_schema()
        return response_model.model_validate(
            {
                "rationale": "Inspect both independent experiments concurrently.",
                "actions": [
                    {"tool": "get_experiment", "experiment_id": "experiment-a"},
                    {"tool": "get_experiment", "experiment_id": "experiment-b"},
                ],
            }
        )


async def test_instructor_reasoning_accepts_multiple_typed_actions(
    monkeypatch: Any,
) -> None:
    """Build and parse a bounded list of discriminated source-tool calls."""
    client = BatchResponseClient()
    monkeypatch.setattr(instructor, "from_provider", lambda *args, **kwargs: client)
    source = AdaptyvDataSource(MockFoundryClient.from_fixture())
    source_state = {
        **initial_state(),
        "phase": "active",
        "objective": {
            "kind": "monitor",
            "resource_ids": ["experiment-a", "experiment-b"],
            "summary": "Monitor both experiments.",
        },
        "evidence": {
            "experiment-a": {"status": "InProduction"},
            "experiment-b": {"status": "InQueue"},
        },
    }
    engine = InstructorAutonomousReasoningEngine(
        "openai/test-model", source, max_wait_seconds=3_600
    )

    turn = await engine.decide([], source_state)

    assert [action.tool_name for action in turn.actions] == [
        "get_experiment",
        "get_experiment",
    ]
    assert client.response_schema is not None
    assert "maxItems" not in client.response_schema["properties"]["actions"]


def test_pending_alert_schema_accepts_only_the_queue_head() -> None:
    """Encode ordered result delivery in the dynamic tool schema before execution."""
    source = AdaptyvDataSource(MockFoundryClient.from_fixture())
    source_state = {
        **initial_state(),
        "phase": "active",
        "pending_alert_resource_ids": ["experiment-a", "experiment-b"],
    }

    [send_alert_type] = available_tool_types(source, source_state)
    schema = send_alert_type.model_json_schema()

    assert schema["properties"]["resource_ids"]["items"]["const"] == "experiment-a"
    assert schema["properties"]["resource_ids"]["maxItems"] == 1


def test_wait_schema_exposes_the_runtime_ceiling() -> None:
    """Tell the model the largest delay that runtime policy will accept."""
    source = AdaptyvDataSource(MockFoundryClient.from_fixture())

    tool_types = available_tool_types(source, initial_state(), max_wait_seconds=900)
    wait_type = next(
        tool_type for tool_type in tool_types if tool_type.model_fields["tool"].default == "wait"
    )

    assert wait_type.model_json_schema()["properties"]["seconds"]["maximum"] == 900
