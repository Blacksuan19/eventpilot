"""Own deterministic monitoring state transitions used by the agent graph."""

from copy import deepcopy
from typing import Any, Literal

from pydantic import Field

from eventpilot.sources.base import DataSource, SourceEffect, SourceExecution, SourceToolCall

ObjectiveKind = Literal["monitor", "report_results", "status_digest", "investigate_incident"]


class SelectObjective(SourceToolCall):
    """Commit a cycle to a validated portfolio and objective type."""

    tool: Literal["select_objective"] = "select_objective"
    kind: ObjectiveKind = Field(
        description=(
            "Cycle intent: monitor follows all active discovered resources; report_results "
            "delivers available results; status_digest summarizes multiple resources; "
            "investigate_incident examines related abnormal resources."
        )
    )
    resource_ids: list[str] = Field(
        min_length=1,
        description=(
            "Discovered platform resource identifiers in scope. A monitor objective must include "
            "every active discovered resource."
        ),
    )
    summary: str = Field(min_length=1, description="Purpose and scope of this objective.")


def initial_state() -> dict[str, Any]:
    """Create empty durable state for one monitored platform."""
    return {
        "phase": "discovery",
        "completed_resource_ids": [],
        "monitoring": {},
        "evidence": {},
        "objective": None,
        "objective_waited": False,
        "poll_interval_seconds": None,
        "last_inspected_resource_id": None,
        "next_resource_candidates": [],
        "pending_alert_resource_id": None,
    }


def available_source_tools(source: DataSource, state: dict[str, Any]) -> set[str]:
    """Expose source tools valid in the graph's current monitoring phase."""
    if state.get("pending_alert_resource_id"):
        return set()
    phase = state.get("phase", "discovery")
    if phase == "discovery":
        return {source.discovery_tool}
    if phase != "active":
        return set()
    evidence = state.get("evidence", {})
    scoped_ids = (state.get("objective") or {}).get("resource_ids", [])
    required = set(required_source_actions(state).values())
    return {
        str(tool_type.model_fields["tool"].default)
        for tool_type in source.tool_types
        if tool_type.model_fields["tool"].default != source.discovery_tool
        and (not required or str(tool_type.model_fields["tool"].default) in required)
        and _tool_is_available(tool_type, scoped_ids, evidence)
    }


def required_source_actions(state: dict[str, Any]) -> dict[str, str]:
    """Return unresolved source operations keyed by objective resource identifier."""
    objective = state.get("objective") or {}
    completed = set(state.get("completed_resource_ids", []))
    evidence = state.get("evidence", {})
    return {
        resource_id: str(required_action)
        for resource_id in objective.get("resource_ids", [])
        if resource_id not in completed
        if (required_action := evidence.get(resource_id, {}).get("required_action"))
    }


def apply_execution(
    execution: SourceExecution, state: dict[str, Any], *, observed_at: float
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reduce normalized source effects into durable graph-owned state."""
    updated = deepcopy(state)
    result = deepcopy(execution.result)
    for effect in execution.effects:
        if effect.kind == "discovery":
            result = _apply_discovery(effect, result, updated, observed_at)
        else:
            _apply_observation(effect, updated, observed_at)
    return result, updated


def select_objective(
    action: SelectObjective, state: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate a graph objective against the latest normalized discovery."""
    updated = deepcopy(state)
    evidence = updated.get("evidence", {})
    discovered_ids = set(evidence)
    selected_ids = set(action.resource_ids)
    rejection = None
    if len(selected_ids) != len(action.resource_ids):
        rejection = "Objective resource identifiers must be unique."
    elif not selected_ids.issubset(discovered_ids):
        rejection = "Objective contains a resource absent from discovery."
    elif action.kind == "monitor":
        monitorable_ids = {
            resource_id
            for resource_id, item in evidence.items()
            if item.get("active", True)
            and resource_id not in set(updated.get("completed_resource_ids", []))
        }
        if selected_ids != monitorable_ids:
            rejection = (
                "A monitor objective must include every active discovered resource so "
                "monitoring can be interleaved."
            )
    elif action.kind == "status_digest" and len(selected_ids) < 2:
        rejection = "A status digest requires at least two resources."
    if rejection:
        return {"status": "rejected", "reason": rejection}, updated
    objective = action.model_dump(mode="json", exclude={"tool"})
    updated.update(
        objective=objective,
        phase="active",
        objective_waited=False,
        poll_interval_seconds=None,
        last_inspected_resource_id=None,
        next_resource_candidates=[],
    )
    return objective, updated


def validate_source_action(action: SourceToolCall, state: dict[str, Any]) -> str | None:
    """Reject scoped source actions that violate the active portfolio rotation."""
    resource_ids = action_resource_ids(action)
    if not resource_ids:
        return None
    objective = state.get("objective") or {}
    scoped_ids = set(objective.get("resource_ids", []))
    outside = [resource_id for resource_id in resource_ids if resource_id not in scoped_ids]
    if outside:
        return "Resource is outside objective scope."
    evidence = state.get("evidence", {})
    required_actions = required_source_actions(state)
    mismatched = [
        resource_id
        for resource_id in resource_ids
        if (required := required_actions.get(resource_id)) and action.tool_name != required
    ]
    if mismatched:
        resource_id = mismatched[0]
        return f"Resource {resource_id} currently requires {required_actions[resource_id]}."
    if resource_ids and not all(
        _tool_requirements_are_met(type(action), evidence.get(resource_id, {}))
        for resource_id in resource_ids
    ):
        return "Resource does not satisfy the selected tool's current requirements."
    candidates = state.get("next_resource_candidates", [])
    if candidates and resource_ids[0] == state.get("last_inspected_resource_id"):
        return (
            "Inspect another portfolio resource before returning to this one. "
            f"Available candidates: {', '.join(candidates)}."
        )
    return None


def action_resource_ids(action: SourceToolCall) -> list[str]:
    """Extract conventional singular resource identifiers from a source action."""
    payload = action.model_dump(mode="json")
    return [str(value) for key, value in payload.items() if key.endswith("_id") and value]


def record_rejected_action(action: SourceToolCall, state: dict[str, Any]) -> dict[str, Any]:
    """Resolve a required operation after an operator explicitly rejects it."""
    updated = deepcopy(state)
    evidence = updated.get("evidence", {})
    for resource_id in action_resource_ids(action):
        resource_evidence = evidence.get(resource_id, {})
        if resource_evidence.get("required_action") == action.tool_name:
            resource_evidence.pop("required_action", None)
            resource_evidence.pop("wait_blocker", None)
            resource_evidence["last_rejected_action"] = action.tool_name
    return updated


def validate_wait(state: dict[str, Any]) -> str | None:
    """Prevent waiting while normalized evidence requires immediate work."""
    pending = state.get("pending_alert_resource_id")
    if pending:
        return f"Result evidence for {pending} must be reported before waiting."
    blockers = [
        str(evidence["wait_blocker"])
        for evidence in state.get("evidence", {}).values()
        if evidence.get("wait_blocker")
    ]
    return blockers[0] if blockers else None


def after_wait(state: dict[str, Any], *, requested_seconds: int, wake_at: float) -> dict[str, Any]:
    """Advance graph scheduling state after the generic wait node completes."""
    updated = deepcopy(state)
    was_idle = updated.get("phase") == "idle"
    objective = updated.get("objective") or {}
    last_inspected = updated.get("last_inspected_resource_id")
    if last_inspected:
        monitoring = updated.setdefault("monitoring", {})
        monitoring.setdefault(last_inspected, {})["next_poll_at"] = wake_at
    completed = set(updated.get("completed_resource_ids", []))
    updated["next_resource_candidates"] = [
        resource_id
        for resource_id in objective.get("resource_ids", [])
        if resource_id != last_inspected and resource_id not in completed
    ]
    updated["objective_waited"] = True
    updated["poll_interval_seconds"] = requested_seconds
    if was_idle:
        updated["phase"] = "discovery"
    return updated


def validate_alert(resource_ids: list[str], state: dict[str, Any]) -> str | None:
    """Require scoped, current normalized evidence before delivery."""
    pending = state.get("pending_alert_resource_id")
    if pending and resource_ids != [pending]:
        return f"Report the pending result for {pending} in its own alert."
    objective = state.get("objective") or {}
    if not set(resource_ids).issubset(set(objective.get("resource_ids", []))):
        return "Resource is outside objective scope."
    evidence = state.get("evidence", {})
    ready = {
        resource_id: bool(evidence.get(resource_id, {}).get("result_ready"))
        for resource_id in resource_ids
    }
    if len(resource_ids) > 1 and any(ready.values()):
        return "Report each resource's results in a separate alert."
    if objective.get("kind") == "report_results" and not all(ready.values()):
        return "Result reports require evidence for every resource."
    if (
        objective.get("kind") == "monitor"
        and not any(ready.values())
        and not state.get("objective_waited", False)
    ):
        return "Active monitoring requires a polling wait before reporting."
    monitoring = state.get("monitoring", {})
    if objective.get("kind") == "monitor" and not any(ready.values()):
        unchanged = [
            resource_id
            for resource_id in resource_ids
            if monitoring.get(resource_id, {}).get("last_reported_status")
            == evidence.get(resource_id, {}).get("status")
        ]
        if unchanged:
            return "An unchanged monitor status was already reported."
    return None


def record_alert(
    resource_ids: list[str], state: dict[str, Any], *, delivered_at: float
) -> dict[str, Any]:
    """Record delivery and prepare either continued work or fresh discovery."""
    updated = deepcopy(state)
    evidence = updated.get("evidence", {})
    completed = updated.setdefault("completed_resource_ids", [])
    newly_completed = [
        resource_id
        for resource_id in resource_ids
        if evidence.get(resource_id, {}).get("result_ready")
    ]
    updated["completed_resource_ids"] = list(dict.fromkeys([*completed, *newly_completed]))
    monitoring = updated.setdefault("monitoring", {})
    next_interval = updated.get("poll_interval_seconds")
    for resource_id in resource_ids:
        record = monitoring.setdefault(resource_id, {})
        observed_status = evidence.get(resource_id, {}).get("status")
        if observed_status is not None:
            record["last_reported_status"] = observed_status
        if next_interval is not None:
            record["next_poll_at"] = delivered_at + next_interval
    updated["pending_alert_resource_id"] = None
    if not should_continue_after_alert(updated):
        updated = record_finish(updated)
    return updated


def should_continue_after_alert(state: dict[str, Any]) -> bool:
    """Continue when another objective resource already has inspected results."""
    objective = state.get("objective") or {}
    completed = set(state.get("completed_resource_ids", []))
    evidence = state.get("evidence", {})
    return any(
        resource_id not in completed and evidence.get(resource_id, {}).get("result_ready")
        for resource_id in objective.get("resource_ids", [])
    )


def validate_finish(state: dict[str, Any]) -> str | None:
    """Reject completion while delivery or an idle wait remains outstanding."""
    pending = state.get("pending_alert_resource_id")
    if pending:
        return f"Result evidence for {pending} must be reported before finishing the cycle."
    required = required_source_actions(state)
    if required:
        resource_id, action = next(iter(required.items()))
        return f"Resource {resource_id} requires {action} before finishing the cycle."
    if state.get("phase") == "idle":
        return "An idle source must wait before starting another discovery cycle."
    return None


def record_finish(state: dict[str, Any]) -> dict[str, Any]:
    """Clear bounded objective state before the next discovery cycle."""
    updated = deepcopy(state)
    updated.update(
        objective=None,
        phase="discovery",
        objective_waited=False,
        poll_interval_seconds=None,
        last_inspected_resource_id=None,
        next_resource_candidates=[],
        pending_alert_resource_id=None,
    )
    return updated


def _apply_discovery(
    effect: SourceEffect,
    result: dict[str, Any],
    state: dict[str, Any],
    observed_at: float,
) -> dict[str, Any]:
    """Filter discovery by durable completion and scheduling state."""
    completed = set(state.get("completed_resource_ids", []))
    monitoring = state.get("monitoring", {})
    actionable = [
        resource
        for resource in effect.resources
        if resource.resource_id not in completed
        and resource.active
        and monitoring.get(resource.resource_id, {}).get("next_poll_at", 0) <= observed_at
    ]
    evidence = state.setdefault("evidence", {})
    for resource in actionable:
        evidence.setdefault(resource.resource_id, {}).update(
            status=resource.status,
            results_status=resource.results_status,
            active=resource.active,
            result_ready=resource.result_ready,
            observed_at=observed_at,
        )
    original_count = len(effect.resources)
    result.update(
        items=[dict(resource.payload) for resource in actionable],
        count=len(actionable),
        total=max(0, int(result.get("total", original_count)) - (original_count - len(actionable))),
    )
    state["phase"] = "objective" if actionable else "idle"
    return result


def _apply_observation(effect: SourceEffect, state: dict[str, Any], observed_at: float) -> None:
    """Merge one source observation and update generic portfolio progress."""
    if effect.resource_id is None:
        return
    evidence = state.setdefault("evidence", {}).setdefault(effect.resource_id, {})
    evidence.update(dict(effect.evidence), observed_at=observed_at)
    if effect.wait_blocker is not None:
        evidence["wait_blocker"] = effect.wait_blocker
    if effect.clear_wait_blocker:
        evidence.pop("wait_blocker", None)
    if effect.required_action is not None:
        evidence["required_action"] = effect.required_action
    if effect.clear_required_action:
        evidence.pop("required_action", None)
    if effect.result_ready:
        evidence["result_ready"] = True
        state["pending_alert_resource_id"] = effect.resource_id
    if effect.inspected:
        monitoring = state.setdefault("monitoring", {}).setdefault(effect.resource_id, {})
        monitoring.update(last_checked_at=observed_at, last_observed_status=evidence.get("status"))
        if effect.resource_id != state.get("last_inspected_resource_id"):
            state["next_resource_candidates"] = []
        state["last_inspected_resource_id"] = effect.resource_id


def _tool_is_available(
    tool_type: type[SourceToolCall],
    scoped_ids: list[str],
    evidence: dict[str, dict[str, Any]],
) -> bool:
    """Evaluate a tool's declarative prerequisites against scoped evidence."""
    return any(
        _tool_requirements_are_met(tool_type, item)
        for resource_id in scoped_ids
        if (item := evidence.get(resource_id, {}))
    )


def _tool_requirements_are_met(tool_type: type[SourceToolCall], evidence: dict[str, Any]) -> bool:
    """Evaluate one tool's declarative requirements for one concrete resource."""
    requirement = tool_type.availability
    if requirement is None:
        return True
    structured_requirement = (
        evidence.get("requirements", {}).get(requirement.requirement_key, {})
        if requirement.requirement_key
        else {}
    )
    return (
        (not requirement.statuses or evidence.get("status") in requirement.statuses)
        and all(evidence.get(key) for key in requirement.evidence_keys)
        and (requirement.requirement_key is None or structured_requirement.get("satisfied") is True)
    )
