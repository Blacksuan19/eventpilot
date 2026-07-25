"""Exercise autonomous trajectories against fuzzed Foundry API fixtures."""

import random
from pathlib import Path
from typing import Any

import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from eventpilot.adapters.adaptyv import ExperimentStatus
from eventpilot.core.agent_reasoning import (
    AgentTurn,
    DemoAutonomousReasoningEngine,
    FinishCycle,
    GetExperiment,
    ListExperiments,
    SendUpdate,
    Wait,
)
from eventpilot.core.autonomous import AgentRuntime, build_autonomous_graph
from eventpilot.core.notifications import DeliveryResult, Notification
from eventpilot.simulator import ExperimentScenario, MockFoundryClient, load_scenarios, tool_names


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


class ScriptedAgent:
    """Return a fixed sequence of tool calls for lifecycle routing tests."""

    def __init__(self, turns: list[AgentTurn]) -> None:
        """Store the turns in the order the graph should request them."""
        self._turns = iter(turns)

    async def decide(
        self, transcript: list[dict[str, Any]], completed_experiment_ids: list[str]
    ) -> AgentTurn:
        """Return the next scripted turn without interpreting tool results."""
        return next(self._turns)


async def no_sleep(delay: float) -> None:
    """Complete agent-selected waits immediately during trajectory tests."""


def scenario_with_lifecycle(
    lifecycle: list[ExperimentStatus], *, fixture_index: int = 0, result_delay_reads: int = 0
) -> ExperimentScenario:
    """Clone one stored API scenario with a lifecycle selected by the test."""
    return load_scenarios()[fixture_index].model_copy(
        update={"lifecycle": lifecycle, "result_delay_reads": result_delay_reads}
    )


async def test_agent_discovers_polls_results_and_finishes_cycle() -> None:
    """Prove the LLM control loop chooses the complete autonomous tool trajectory."""
    foundry = MockFoundryClient(
        [
            scenario_with_lifecycle(
                [ExperimentStatus.IN_QUEUE, ExperimentStatus.IN_PRODUCTION, ExperimentStatus.DONE]
            )
        ]
    )
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


async def test_agent_investigates_updates_while_completed_results_are_delayed() -> None:
    """Cover terminal status, unavailable results, update inspection, and a later retry."""
    foundry = MockFoundryClient(
        [scenario_with_lifecycle([ExperimentStatus.DONE], result_delay_reads=1)]
    )
    sink = RecordingSink()
    graph = build_autonomous_graph(DemoAutonomousReasoningEngine(), foundry, sink, sleep=no_sleep)

    result = await AgentRuntime(graph).run(max_cycles=1)

    transcript = result.get("transcript")
    assert transcript is not None
    assert tool_names(transcript) == [
        "list_experiments",
        "get_experiment",
        "list_experiment_updates",
        "wait",
        "get_experiment",
        "list_experiment_results",
        "send_update",
    ]
    assert len(sink.notifications) == 1


async def test_idle_agent_waits_without_sending_an_update() -> None:
    """Cover a discovery cycle that finds no actionable work and backs off cleanly."""
    agent = ScriptedAgent(
        [
            AgentTurn(rationale="Discover work.", action=ListExperiments()),
            AgentTurn(rationale="No work is active.", action=Wait(seconds=60, reason="Idle.")),
            AgentTurn(rationale="Yield after the idle check.", action=FinishCycle(summary="Idle.")),
        ]
    )
    sink = RecordingSink()
    graph = build_autonomous_graph(
        agent,
        MockFoundryClient([scenario_with_lifecycle([ExperimentStatus.DONE])]),
        sink,
        sleep=no_sleep,
    )

    result = await AgentRuntime(graph).run(max_cycles=1)

    assert tool_names(result.get("transcript", [])) == ["list_experiments", "wait"]
    assert result.get("cycle_summary") == "Idle."
    assert sink.notifications == []


async def test_agent_selects_active_experiment_from_mixed_discovery_results() -> None:
    """Prove discovery ignores completed work when an active experiment is available."""
    agent = DemoAutonomousReasoningEngine()
    turn = await agent.decide(
        [
            {
                "tool": "list_experiments",
                "call": {"tool": "list_experiments", "limit": 50, "offset": 0},
                "result": {
                    "items": [
                        {"id": "complete", "status": "Done", "results_status": "All"},
                        {"id": "active", "status": "InProduction", "results_status": "None"},
                    ],
                    "total": 2,
                    "count": 2,
                    "offset": 0,
                },
            }
        ],
        [],
    )

    assert isinstance(turn.action, GetExperiment)
    assert turn.action.experiment_id == "active"


async def test_supervisor_completes_two_experiments_across_fresh_cycles() -> None:
    """Prove the normal API collection is handled across fresh agent cycles."""
    scenarios = [
        scenario_with_lifecycle(
            [ExperimentStatus.IN_QUEUE, ExperimentStatus.DONE], fixture_index=0
        ),
        scenario_with_lifecycle(
            [ExperimentStatus.IN_PRODUCTION, ExperimentStatus.DONE], fixture_index=1
        ),
    ]
    experiment_ids = [scenario.experiment.id for scenario in scenarios]
    foundry = MockFoundryClient(scenarios)
    sink = RecordingSink()
    graph = build_autonomous_graph(DemoAutonomousReasoningEngine(), foundry, sink, sleep=no_sleep)

    result = await AgentRuntime(graph).run(max_cycles=2)

    assert foundry.inspected_ids == [
        experiment_ids[0],
        experiment_ids[0],
        experiment_ids[1],
        experiment_ids[1],
    ]
    assert [notification.title for notification in sink.notifications] == [
        f"Experiment {experiment_ids[0]} results are ready",
        f"Experiment {experiment_ids[1]} results are ready",
    ]
    assert result.get("cycle_count") == 2
    assert result.get("completed_experiment_ids") == experiment_ids


async def test_result_delivery_is_idempotent_across_fresh_cycles() -> None:
    """Prove durable completion state suppresses a repeated result notification."""
    scenario = scenario_with_lifecycle([ExperimentStatus.DONE])
    experiment_id = scenario.experiment.id
    agent = ScriptedAgent(
        [
            AgentTurn(
                rationale="Read completion.", action=GetExperiment(experiment_id=experiment_id)
            ),
            AgentTurn(
                rationale="Report results.",
                action=SendUpdate(
                    experiment_ids=[experiment_id],
                    title="Results ready",
                    body="The result is ready.",
                ),
            ),
            AgentTurn(rationale="Finish.", action=FinishCycle(summary="Reported.")),
            AgentTurn(
                rationale="Read completion again.",
                action=GetExperiment(experiment_id=experiment_id),
            ),
            AgentTurn(
                rationale="Attempt duplicate report.",
                action=SendUpdate(
                    experiment_ids=[experiment_id],
                    title="Results ready",
                    body="The result is ready.",
                ),
            ),
            AgentTurn(rationale="Finish.", action=FinishCycle(summary="Checked.")),
        ]
    )
    sink = RecordingSink()
    graph = build_autonomous_graph(
        agent,
        MockFoundryClient([scenario]),
        sink,
        sleep=no_sleep,
    )

    result = await AgentRuntime(graph).run(max_cycles=2)

    assert len(sink.notifications) == 1
    assert result.get("completed_experiment_ids") == [experiment_id]
    transcript = result.get("transcript", [])
    assert transcript[-1]["result"]["status"] == "skipped"


async def test_combined_update_records_every_completed_experiment() -> None:
    """Prove one multi-experiment update atomically records every evidenced completion."""
    scenarios = [
        scenario_with_lifecycle([ExperimentStatus.DONE], fixture_index=0),
        scenario_with_lifecycle([ExperimentStatus.DONE], fixture_index=1),
    ]
    experiment_ids = [scenario.experiment.id for scenario in scenarios]
    agent = ScriptedAgent(
        [
            AgentTurn(
                rationale="Read A.",
                action=GetExperiment(experiment_id=experiment_ids[0]),
            ),
            AgentTurn(
                rationale="Read B.",
                action=GetExperiment(experiment_id=experiment_ids[1]),
            ),
            AgentTurn(
                rationale="Report both.",
                action=SendUpdate(
                    experiment_ids=experiment_ids,
                    title="Both results are ready",
                    body="Both experiments completed.",
                ),
            ),
            AgentTurn(rationale="Finish.", action=FinishCycle(summary="Reported both.")),
        ]
    )
    sink = RecordingSink()
    foundry = MockFoundryClient(scenarios)
    graph = build_autonomous_graph(agent, foundry, sink, sleep=no_sleep)

    result = await AgentRuntime(graph).run(max_cycles=1)

    assert len(sink.notifications) == 1
    assert result.get("completed_experiment_ids") == experiment_ids


async def test_sqlite_checkpoint_survives_fresh_cycles(tmp_path: Path) -> None:
    """Prove fresh invocations retain durable counters while clearing working context."""
    sink = RecordingSink()
    database = tmp_path / "checkpoints.sqlite"
    first_scenario = scenario_with_lifecycle(
        [ExperimentStatus.IN_QUEUE, ExperimentStatus.DONE], fixture_index=0
    )
    second_scenario = scenario_with_lifecycle(
        [ExperimentStatus.IN_QUEUE, ExperimentStatus.DONE], fixture_index=1
    )
    experiment_ids = [first_scenario.experiment.id, second_scenario.experiment.id]
    async with AsyncSqliteSaver.from_conn_string(str(database)) as checkpointer:
        graph = build_autonomous_graph(
            DemoAutonomousReasoningEngine(),
            MockFoundryClient([first_scenario]),
            sink,
            checkpointer=checkpointer,
            sleep=no_sleep,
        )
        first = await AgentRuntime(graph).run(max_cycles=1)
        restarted_graph = build_autonomous_graph(
            DemoAutonomousReasoningEngine(),
            MockFoundryClient(
                [
                    first_scenario.model_copy(update={"lifecycle": [ExperimentStatus.DONE]}),
                    second_scenario,
                ]
            ),
            sink,
            checkpointer=checkpointer,
            sleep=no_sleep,
        )
        second = await AgentRuntime(restarted_graph).run(max_cycles=1)

    assert first.get("cycle_count") == 1
    assert second.get("cycle_count") == 2
    assert len(second.get("transcript", [])) == 6
    assert second.get("completed_experiment_ids") == experiment_ids


@pytest.mark.parametrize("seed", range(10))
async def test_agent_handles_fuzzed_experiment_collections(seed: int) -> None:
    """Fuzz collection order and lifecycle events through the real graph loop."""
    randomizer = random.Random(seed)
    scenarios = load_scenarios()
    randomizer.shuffle(scenarios)
    intermediate_statuses = [
        ExperimentStatus.IN_QUEUE,
        ExperimentStatus.IN_PRODUCTION,
        ExperimentStatus.DATA_ANALYSIS,
        ExperimentStatus.IN_REVIEW,
    ]
    fuzzed = [
        scenario.model_copy(
            update={
                "lifecycle": [
                    *randomizer.sample(
                        intermediate_statuses,
                        k=randomizer.randint(0, len(intermediate_statuses)),
                    ),
                    ExperimentStatus.DONE,
                ]
            }
        )
        for scenario in scenarios
    ]
    expected_ids = [scenario.experiment.id for scenario in fuzzed]
    sink = RecordingSink()
    graph = build_autonomous_graph(
        DemoAutonomousReasoningEngine(),
        MockFoundryClient(fuzzed),
        sink,
        sleep=no_sleep,
    )

    result = await AgentRuntime(graph).run(max_cycles=len(fuzzed))

    assert set(result.get("completed_experiment_ids", [])) == set(expected_ids)
    assert len(sink.notifications) == len(fuzzed)
    assert all("results are ready" in notification.title for notification in sink.notifications)
