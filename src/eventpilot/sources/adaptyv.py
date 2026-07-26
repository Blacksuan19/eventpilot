"""Expose the Adaptyv Foundry API as a pluggable agent data source."""

from importlib.resources import files
from typing import Any

from eventpilot.adapters.adaptyv import ExperimentStatus, FoundryClient
from eventpilot.adapters.adaptyv.tools import (
    AcceptExperimentQuote,
    GetExperiment,
    GetExperimentQuote,
    ListExperimentResults,
    ListExperiments,
    ListExperimentUpdates,
    SubmitExperiment,
    UpdateExperiment,
)
from eventpilot.core.approvals import ApprovalRequest
from eventpilot.sources.base import (
    ResourceSnapshot,
    SourceContext,
    SourceEffect,
    SourceExecution,
    SourceToolCall,
)


class AdaptyvDataSource:
    """Expose Foundry API operations and normalize their effects for the agent."""

    name = "adaptyv-foundry"
    discovery_tool = "list_experiments"
    tool_types: tuple[type[SourceToolCall], ...] = (
        ListExperiments,
        GetExperiment,
        ListExperimentUpdates,
        ListExperimentResults,
        UpdateExperiment,
        SubmitExperiment,
        AcceptExperimentQuote,
        GetExperimentQuote,
    )

    def __init__(self, client: FoundryClient) -> None:
        """Bind the agent data source to a real or fixture-backed Foundry API client."""
        self._client = client
        self.instructions = (
            files("eventpilot.adapters.adaptyv")
            .joinpath("instructions.txt")
            .read_text(encoding="utf-8")
        )
        self._tool_type_registry = {
            tool_type.model_fields["tool"].default: tool_type for tool_type in self.tool_types
        }

    def parse_tool(self, payload: dict[str, Any]) -> SourceToolCall:
        """Validate a persisted Foundry tool call by its discriminator."""
        tool_type = self._tool_type_registry.get(payload.get("tool"))
        if tool_type is None:
            raise ValueError(f"Unknown {self.name} tool: {payload.get('tool')}")
        return tool_type.model_validate(payload)

    async def execute(self, action: SourceToolCall, context: SourceContext) -> SourceExecution:
        """Execute one Foundry operation and emit normalized monitoring facts."""
        if isinstance(action, ListExperiments):
            return await self._list_experiments(action)
        if isinstance(action, GetExperiment):
            return await self._get_experiment(action)
        if isinstance(action, ListExperimentUpdates):
            return await self._list_experiment_updates(action)
        if isinstance(action, ListExperimentResults):
            return await self._list_experiment_results(action)
        if isinstance(action, UpdateExperiment):
            return await self._update_experiment(action, context)
        if isinstance(action, SubmitExperiment):
            return await self._submit_experiment(action, context)
        if isinstance(action, AcceptExperimentQuote):
            return await self._accept_experiment_quote(action, context)
        if isinstance(action, GetExperimentQuote):
            return await self._get_experiment_quote(action, context)
        raise TypeError(f"Unsupported {self.name} action: {type(action).__name__}")

    def approval_request(
        self, action: SourceToolCall, state: dict[str, Any]
    ) -> ApprovalRequest | None:
        """Require operator confirmation before quote acceptance creates an invoice."""
        if not isinstance(action, AcceptExperimentQuote):
            return None
        evidence = state.get("evidence", {}).get(action.experiment_id, {})
        quote = evidence.get("quote", {})
        amount = quote.get("amount_total")
        currency = str(quote.get("currency", "")).upper()
        formatted_amount = (
            f"{currency} {amount / 100:,.2f}" if isinstance(amount, int) else "unknown amount"
        )
        return ApprovalRequest(
            title=f"Approve quote for {action.experiment_id}",
            body=(
                f"Foundry quoted {formatted_amount} for {action.experiment_id}. "
                "Approval will accept the quote and create an invoice."
            ),
            resource_ids=(action.experiment_id,),
        )

    async def _list_experiments(self, action: ListExperiments) -> SourceExecution:
        """Discover Foundry experiments and normalize their lifecycle state."""
        page = await self._client.list_experiments(limit=action.limit, offset=action.offset)
        result = page.model_dump(mode="json")
        resources = tuple(
            ResourceSnapshot(
                resource_id=item["id"],
                status=item["status"],
                results_status=item["results_status"],
                active=item["status"] != ExperimentStatus.CANCELED,
                result_ready=(
                    item["status"] == ExperimentStatus.DONE
                    and item["results_status"] in {"Partial", "All"}
                ),
                payload=item,
            )
            for item in result["items"]
        )
        return SourceExecution(result=result, effects=(SourceEffect("discovery", resources),))

    async def _get_experiment(self, action: GetExperiment) -> SourceExecution:
        """Fetch experiment detail and normalize current status evidence."""
        experiment = await self._client.get_experiment(action.experiment_id)
        result = experiment.model_dump(mode="json")
        blocker = None
        if result["status"] == ExperimentStatus.DRAFT:
            blocker = f"Draft {action.experiment_id} must be prepared and submitted before waiting."
        elif result["status"] == ExperimentStatus.QUOTE_SENT:
            blocker = f"Quote {action.experiment_id} must be reviewed for approval before waiting."
        effect = SourceEffect(
            "observation",
            resource_id=action.experiment_id,
            evidence={
                "status": result["status"],
                "results_status": result["results_status"],
                "result_ready": (
                    result["status"] == ExperimentStatus.DONE
                    and result["results_status"] in {"Partial", "All"}
                ),
                "experiment_spec": result.get("experiment_spec", {}),
                "detail_observed": True,
            },
            inspected=True,
            wait_blocker=blocker,
            clear_wait_blocker=blocker is None,
        )
        return SourceExecution(result=result, effects=(effect,))

    async def _update_experiment(
        self, action: UpdateExperiment, context: SourceContext
    ) -> SourceExecution:
        """Modify an editable experiment after validating observed Foundry state."""
        evidence = self._evidence(context, action.experiment_id)
        if evidence.get("status") not in {ExperimentStatus.DRAFT, ExperimentStatus.IN_REVIEW}:
            return self._rejected("Experiment is not editable.")
        experiment = await self._client.update_experiment(action.experiment_id, action.changes)
        result = experiment.model_dump(mode="json")
        effect = SourceEffect(
            "observation",
            resource_id=action.experiment_id,
            evidence={
                "status": result["status"],
                "experiment_spec": result.get("experiment_spec", {}),
                "last_action": action.tool_name,
            },
            inspected=True,
            wait_blocker=(
                f"Draft {action.experiment_id} must be submitted before waiting."
                if result["status"] == ExperimentStatus.DRAFT
                else None
            ),
        )
        return SourceExecution(result=result, effects=(effect,))

    async def _submit_experiment(
        self, action: SubmitExperiment, context: SourceContext
    ) -> SourceExecution:
        """Submit a valid draft after enforcing the demonstration replicate policy."""
        evidence = self._evidence(context, action.experiment_id)
        if evidence.get("status") != ExperimentStatus.DRAFT:
            return self._rejected("Experiment is not a draft.")
        if evidence.get("experiment_spec", {}).get("n_replicates", 0) < 2:
            return self._rejected(
                "Draft policy requires at least two replicates before submission."
            )
        confirmation = await self._client.submit_experiment(action.experiment_id)
        result = confirmation.model_dump(mode="json")
        effect = SourceEffect(
            "observation",
            resource_id=action.experiment_id,
            evidence={"status": result["status"], "last_action": action.tool_name},
            inspected=True,
            clear_wait_blocker=True,
        )
        return SourceExecution(result=result, effects=(effect,))

    async def _get_experiment_quote(
        self, action: GetExperimentQuote, context: SourceContext
    ) -> SourceExecution:
        """Read quote cost and expiry before allowing consequential acceptance."""
        evidence = self._evidence(context, action.experiment_id)
        if evidence.get("status") != ExperimentStatus.QUOTE_SENT:
            return self._rejected("Experiment has no pending quote.")
        quote = await self._client.get_experiment_quote(action.experiment_id)
        result = quote.model_dump(mode="json")
        effect = SourceEffect(
            "observation",
            resource_id=action.experiment_id,
            evidence={"quote": result, "quote_observed": True},
            inspected=True,
            wait_blocker=(
                f"Quote {action.experiment_id} requires an operator decision before waiting."
            ),
        )
        return SourceExecution(result=result, effects=(effect,))

    async def _accept_experiment_quote(
        self, action: AcceptExperimentQuote, context: SourceContext
    ) -> SourceExecution:
        """Accept an approved quote and normalize its invoice transition."""
        evidence = self._evidence(context, action.experiment_id)
        if evidence.get("status") != ExperimentStatus.QUOTE_SENT:
            return self._rejected("Experiment has no pending quote.")
        confirmation = await self._client.accept_experiment_quote(action.experiment_id)
        result = confirmation.model_dump(mode="json")
        effect = SourceEffect(
            "observation",
            resource_id=action.experiment_id,
            evidence={
                "status": ExperimentStatus.WAITING_FOR_MATERIALS.value,
                "last_action": action.tool_name,
                "invoice_id": result.get("invoice_id"),
            },
            inspected=True,
            clear_wait_blocker=True,
        )
        return SourceExecution(result=result, effects=(effect,))

    async def _list_experiment_updates(self, action: ListExperimentUpdates) -> SourceExecution:
        """Fetch experiment updates and record endpoint inspection."""
        page = await self._client.list_experiment_updates(action.experiment_id)
        result = page.model_dump(mode="json")
        effect = SourceEffect(
            "observation",
            resource_id=action.experiment_id,
            evidence={"update_count": result["count"], "updates_observed": True},
            inspected=True,
        )
        return SourceExecution(result=result, effects=(effect,))

    async def _list_experiment_results(self, action: ListExperimentResults) -> SourceExecution:
        """Fetch experiment results and mark inspected operator-ready data."""
        page = await self._client.list_experiment_results(action.experiment_id)
        result = page.model_dump(mode="json")
        effect = SourceEffect(
            "observation",
            resource_id=action.experiment_id,
            evidence={"result_count": result["count"], "results_observed": True},
            inspected=True,
            result_ready=result["count"] > 0,
        )
        return SourceExecution(result=result, effects=(effect,))

    @staticmethod
    def _evidence(context: SourceContext, resource_id: str) -> dict[str, Any]:
        """Return the graph's latest normalized evidence for one experiment."""
        return context.state.get("evidence", {}).get(resource_id, {})

    @staticmethod
    def _rejected(reason: str) -> SourceExecution:
        """Return a side-effect-free platform precondition rejection."""
        return SourceExecution(result={"status": "rejected", "reason": reason})
