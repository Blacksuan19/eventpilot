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
    ListExperimentResults,
    ListExperiments,
    SelectObjective,
    SendUpdate,
    Wait,
)
from eventpilot.core.autonomous import AgentRuntime, build_autonomous_graph
from eventpilot.core.notifications import DeliveryResult, Notification
from eventpilot.simulator import (
    AcceleratedClock,
    ExperimentScenario,
    LifecycleStep,
    MockFoundryClient,
    load_scenarios,
    tool_names,
)


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
        self,
        transcript: list[dict[str, Any]],
        completed_experiment_ids: list[str],
        monitoring: dict[str, dict[str, Any]],
        evidence: dict[str, dict[str, Any]],
    ) -> AgentTurn:
        """Return the next scripted turn without interpreting tool results."""
        return next(self._turns)


class ManualClock:
    """Advance fixture time only when the agent executes its wait tool."""

    def __init__(self) -> None:
        """Start the deterministic clock at zero seconds."""
        self.now = 0.0
        self.waits: list[float] = []

    def __call__(self) -> float:
        """Return the current deterministic monotonic time."""
        return self.now

    async def sleep(self, delay: float) -> None:
        """Record an agent-selected delay and advance time without blocking tests."""
        self.waits.append(delay)
        self.now += delay


async def test_accelerated_clock_advances_logical_time_without_real_delay() -> None:
    """Prove Docker acceleration advances the shared hidden clock by its multiplier."""
    clock = AcceleratedClock(3_600)
    started_at = clock()

    await clock.sleep(2)

    assert clock() - started_at == 7_200


def scenario_with_lifecycle(
    lifecycle: list[ExperimentStatus], *, fixture_index: int = 0, done_delay_seconds: int = 0
) -> ExperimentScenario:
    """Clone one stored scenario with short deterministic timed lifecycle steps."""
    steps = [
        LifecycleStep(
            status=status,
            duration_seconds=(done_delay_seconds if status == ExperimentStatus.DONE else 1),
        )
        for status in lifecycle
    ]
    return load_scenarios()[fixture_index].model_copy(update={"lifecycle": steps})


def timed_foundry(scenarios: list[ExperimentScenario]) -> tuple[MockFoundryClient, ManualClock]:
    """Bind fixture scenarios and graph sleeps to one deterministic clock."""
    clock = ManualClock()
    return MockFoundryClient(scenarios, clock=clock), clock


def fuzz_lifecycle(randomizer: random.Random) -> list[LifecycleStep]:
    """Generate a terminating timed lifecycle with hidden randomized stage durations."""
    candidates = [
        ExperimentStatus.IN_QUEUE,
        ExperimentStatus.IN_PRODUCTION,
        ExperimentStatus.DATA_ANALYSIS,
        ExperimentStatus.IN_REVIEW,
    ]
    statuses = randomizer.sample(candidates, k=randomizer.randint(0, len(candidates)))
    return [
        *[
            LifecycleStep(status=status, duration_seconds=randomizer.randint(1, 3))
            for status in statuses
        ],
        LifecycleStep(status=ExperimentStatus.DONE, duration_seconds=randomizer.randint(0, 2)),
    ]


async def test_agent_discovers_polls_results_and_finishes_cycle() -> None:
    """Prove the LLM control loop chooses the complete autonomous tool trajectory."""
    foundry, clock = timed_foundry(
        [
            scenario_with_lifecycle(
                [ExperimentStatus.IN_QUEUE, ExperimentStatus.IN_PRODUCTION, ExperimentStatus.DONE]
            )
        ]
    )
    sink = RecordingSink()
    graph = build_autonomous_graph(
        DemoAutonomousReasoningEngine(), foundry, sink, sleep=clock.sleep, clock=clock
    )

    result = await AgentRuntime(graph).run(max_cycles=1)

    transcript = result.get("transcript")
    assert transcript is not None
    assert tool_names(transcript) == [
        "list_experiments",
        "select_objective",
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
    assert clock.waits == [1, 1]


async def test_budget_yield_can_finish_after_discovery() -> None:
    """Allow a bounded cycle to yield after discovery before choosing an objective."""
    foundry, clock = timed_foundry([scenario_with_lifecycle([ExperimentStatus.IN_QUEUE])])
    agent = ScriptedAgent(
        [
            AgentTurn(
                rationale="Discover current experiments.",
                action=ListExperiments(),
            ),
            AgentTurn(
                rationale="Yield after reaching this cycle's budget.",
                action=FinishCycle(summary="Resume from discovery in a fresh cycle."),
            ),
        ]
    )
    graph = build_autonomous_graph(
        agent,
        foundry,
        RecordingSink(),
        sleep=clock.sleep,
        clock=clock,
        max_tool_calls_per_cycle=1,
    )

    result = await AgentRuntime(graph).run(max_cycles=1)

    assert result.get("outcome") == "cycle_finished"
    assert result.get("cycle_summary") == "Resume from discovery in a fresh cycle."
    assert tool_names(result.get("transcript", [])) == ["list_experiments"]


async def test_agent_investigates_updates_while_completed_results_are_delayed() -> None:
    """Cover terminal status, unavailable results, update inspection, and a later retry."""
    foundry, clock = timed_foundry(
        [scenario_with_lifecycle([ExperimentStatus.DONE], done_delay_seconds=1)]
    )
    sink = RecordingSink()
    graph = build_autonomous_graph(
        DemoAutonomousReasoningEngine(), foundry, sink, sleep=clock.sleep, clock=clock
    )

    result = await AgentRuntime(graph).run(max_cycles=1)

    transcript = result.get("transcript")
    assert transcript is not None
    assert tool_names(transcript) == [
        "list_experiments",
        "select_objective",
        "get_experiment",
        "list_experiment_updates",
        "wait",
        "get_experiment",
        "list_experiment_results",
        "send_update",
    ]
    assert len(sink.notifications) == 1
    assert clock.waits == [5]


async def test_fixture_time_advances_only_with_the_hidden_clock() -> None:
    """Prove API reads cannot advance stages and timing metadata never reaches the agent."""
    scenario = load_scenarios()[0].model_copy(
        update={
            "lifecycle": [
                LifecycleStep(status=ExperimentStatus.IN_QUEUE, duration_seconds=10),
                LifecycleStep(status=ExperimentStatus.DONE, duration_seconds=5),
            ]
        }
    )
    foundry, clock = timed_foundry([scenario])
    experiment_id = scenario.experiment.id

    discovery = await foundry.list_experiments()
    first = await foundry.get_experiment(experiment_id)
    repeated = await foundry.get_experiment(experiment_id)
    await clock.sleep(10)
    done_without_results = await foundry.get_experiment(experiment_id)
    await clock.sleep(5)
    done_with_results = await foundry.get_experiment(experiment_id)

    assert first.status == repeated.status == ExperimentStatus.IN_QUEUE
    assert done_without_results.status == ExperimentStatus.DONE
    assert done_without_results.results_status == "None"
    assert done_with_results.results_status == "All"
    assert "duration_seconds" not in discovery.model_dump_json()


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
    foundry, clock = timed_foundry([scenario_with_lifecycle([ExperimentStatus.DONE])])
    graph = build_autonomous_graph(
        agent,
        foundry,
        sink,
        sleep=clock.sleep,
        clock=clock,
    )

    result = await graph.ainvoke(
        {
            "transcript": [],
            "phase": "discovery",
            "completed_experiment_ids": [load_scenarios()[0].experiment.id],
        },
        config={"configurable": {"thread_id": "idle-test"}},
    )

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
        {},
        {},
    )

    assert isinstance(turn.action, SelectObjective)
    assert turn.action.experiment_ids == ["active"]


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
    foundry, clock = timed_foundry(scenarios)
    sink = RecordingSink()
    graph = build_autonomous_graph(
        DemoAutonomousReasoningEngine(), foundry, sink, sleep=clock.sleep, clock=clock
    )

    result = await AgentRuntime(graph).run(max_cycles=2)

    assert foundry.inspected_ids == [
        experiment_ids[0],
        experiment_ids[0],
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
            AgentTurn(rationale="Discover.", action=ListExperiments()),
            AgentTurn(
                rationale="Scope result delivery.",
                action=SelectObjective(
                    kind="report_results",
                    experiment_ids=[experiment_id],
                    summary="Report available results.",
                ),
            ),
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
            AgentTurn(rationale="Discover again.", action=ListExperiments()),
            AgentTurn(rationale="Nothing remains.", action=Wait(seconds=1, reason="Idle.")),
            AgentTurn(rationale="Finish idle cycle.", action=FinishCycle(summary="Checked.")),
        ]
    )
    sink = RecordingSink()
    foundry, clock = timed_foundry([scenario])
    graph = build_autonomous_graph(
        agent,
        foundry,
        sink,
        sleep=clock.sleep,
        clock=clock,
    )

    result = await AgentRuntime(graph).run(max_cycles=2)

    assert len(sink.notifications) == 1
    assert result.get("completed_experiment_ids") == [experiment_id]
    assert tool_names(result.get("transcript", [])) == ["list_experiments", "wait"]


async def test_combined_update_records_every_completed_experiment() -> None:
    """Prove one multi-experiment update atomically records every evidenced completion."""
    scenarios = [
        scenario_with_lifecycle([ExperimentStatus.DONE], fixture_index=0),
        scenario_with_lifecycle([ExperimentStatus.DONE], fixture_index=1),
    ]
    experiment_ids = [scenario.experiment.id for scenario in scenarios]
    agent = ScriptedAgent(
        [
            AgentTurn(rationale="Discover.", action=ListExperiments()),
            AgentTurn(
                rationale="Group related completed results.",
                action=SelectObjective(
                    kind="report_results",
                    experiment_ids=experiment_ids,
                    summary="Report both completed experiments.",
                ),
            ),
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
        ]
    )
    sink = RecordingSink()
    foundry, clock = timed_foundry(scenarios)
    graph = build_autonomous_graph(agent, foundry, sink, sleep=clock.sleep, clock=clock)

    result = await AgentRuntime(graph).run(max_cycles=1)

    assert len(sink.notifications) == 1
    assert result.get("completed_experiment_ids") == experiment_ids


async def test_graph_rejects_experiment_calls_outside_objective_scope() -> None:
    """Prove an LLM cannot inspect an experiment outside its committed objective."""
    scenarios = [
        scenario_with_lifecycle([ExperimentStatus.DONE], fixture_index=0),
        scenario_with_lifecycle([ExperimentStatus.DONE], fixture_index=1),
    ]
    experiment_ids = [scenario.experiment.id for scenario in scenarios]
    agent = ScriptedAgent(
        [
            AgentTurn(rationale="Discover.", action=ListExperiments()),
            AgentTurn(
                rationale="Monitor only the first experiment.",
                action=SelectObjective(
                    kind="monitor",
                    experiment_ids=[experiment_ids[0]],
                    summary="Monitor the first experiment.",
                ),
            ),
            AgentTurn(
                rationale="Try unrelated work.",
                action=GetExperiment(experiment_id=experiment_ids[1]),
            ),
            AgentTurn(
                rationale="Return to scoped work.",
                action=GetExperiment(experiment_id=experiment_ids[0]),
            ),
            AgentTurn(
                rationale="Read scoped results.",
                action=ListExperimentResults(experiment_id=experiment_ids[0]),
            ),
            AgentTurn(
                rationale="Report scoped results.",
                action=SendUpdate(
                    experiment_ids=[experiment_ids[0]],
                    title="Scoped result",
                    body="The scoped result is ready.",
                ),
            ),
        ]
    )
    foundry, clock = timed_foundry(scenarios)
    sink = RecordingSink()
    graph = build_autonomous_graph(agent, foundry, sink, sleep=clock.sleep, clock=clock)

    result = await AgentRuntime(graph).run(max_cycles=1)

    rejected = result.get("transcript", [])[2]
    assert rejected["result"]["status"] == "rejected"
    assert "outside" in rejected["result"]["reason"]
    assert foundry.inspected_ids == [experiment_ids[0]]
    assert len(sink.notifications) == 1


async def test_graph_rejects_multi_experiment_monitor_objective() -> None:
    """Prove monitoring cannot silently expand into an unrelated experiment batch."""
    scenarios = [
        scenario_with_lifecycle([ExperimentStatus.DONE], fixture_index=0),
        scenario_with_lifecycle([ExperimentStatus.DONE], fixture_index=1),
    ]
    experiment_ids = [scenario.experiment.id for scenario in scenarios]
    agent = ScriptedAgent(
        [
            AgentTurn(rationale="Discover.", action=ListExperiments()),
            AgentTurn(
                rationale="Invalid broad monitor.",
                action=SelectObjective(
                    kind="monitor",
                    experiment_ids=experiment_ids,
                    summary="Monitor everything.",
                ),
            ),
            AgentTurn(
                rationale="Use a valid result batch.",
                action=SelectObjective(
                    kind="report_results",
                    experiment_ids=experiment_ids,
                    summary="Report related ready results.",
                ),
            ),
            AgentTurn(
                rationale="Inspect first.",
                action=GetExperiment(experiment_id=experiment_ids[0]),
            ),
            AgentTurn(
                rationale="Inspect second.",
                action=GetExperiment(experiment_id=experiment_ids[1]),
            ),
            AgentTurn(
                rationale="Report both.",
                action=SendUpdate(
                    experiment_ids=experiment_ids,
                    title="Related results",
                    body="Both results are ready.",
                ),
            ),
        ]
    )
    sink = RecordingSink()
    foundry, clock = timed_foundry(scenarios)
    graph = build_autonomous_graph(agent, foundry, sink, sleep=clock.sleep, clock=clock)

    result = await AgentRuntime(graph).run(max_cycles=1)

    rejected = result.get("transcript", [])[1]
    assert rejected["result"]["status"] == "rejected"
    assert "exactly one" in rejected["result"]["reason"]
    assert len(sink.notifications) == 1


async def test_active_monitor_requires_agent_selected_wait_before_reporting() -> None:
    """Prove an active monitoring objective cannot skip its longitudinal poll."""
    scenario = load_scenarios()[0].model_copy(
        update={
            "lifecycle": [
                LifecycleStep(status=ExperimentStatus.IN_QUEUE, duration_seconds=60),
                LifecycleStep(status=ExperimentStatus.DONE, duration_seconds=0),
            ]
        }
    )
    experiment_id = scenario.experiment.id
    agent = ScriptedAgent(
        [
            AgentTurn(rationale="Discover.", action=ListExperiments()),
            AgentTurn(
                rationale="Monitor the active experiment.",
                action=SelectObjective(
                    kind="monitor",
                    experiment_ids=[experiment_id],
                    summary="Monitor queue progress.",
                ),
            ),
            AgentTurn(
                rationale="Inspect current state.",
                action=GetExperiment(experiment_id=experiment_id),
            ),
            AgentTurn(
                rationale="Try to report immediately.",
                action=SendUpdate(
                    experiment_ids=[experiment_id],
                    title="Still queued",
                    body="The experiment remains queued.",
                ),
            ),
            AgentTurn(
                rationale="Poll later.",
                action=Wait(seconds=30, reason="Check queue progress later."),
            ),
            AgentTurn(rationale="Attempt forbidden rediscovery.", action=ListExperiments()),
            AgentTurn(
                rationale="Refresh after waiting.",
                action=GetExperiment(experiment_id=experiment_id),
            ),
            AgentTurn(
                rationale="Report longitudinal status.",
                action=SendUpdate(
                    experiment_ids=[experiment_id],
                    title="Queue monitoring update",
                    body="The experiment remains queued after 30 seconds.",
                ),
            ),
            AgentTurn(rationale="Start a fresh discovery.", action=ListExperiments()),
            AgentTurn(
                rationale="Nothing is eligible yet.",
                action=Wait(seconds=30, reason="Wait until the next eligible poll."),
            ),
            AgentTurn(
                rationale="End the idle cycle.", action=FinishCycle(summary="No eligible work.")
            ),
        ]
    )
    sink = RecordingSink()
    foundry, clock = timed_foundry([scenario])
    graph = build_autonomous_graph(agent, foundry, sink, sleep=clock.sleep, clock=clock)

    result = await AgentRuntime(graph).run(max_cycles=2)

    monitoring = result.get("monitoring", {})[experiment_id]
    assert monitoring == {
        "last_checked_at": 30.0,
        "last_observed_status": "InQueue",
        "last_reported_status": "InQueue",
        "next_poll_at": 60.0,
    }
    transcript = result.get("transcript", [])
    assert transcript[0]["result"]["items"] == []
    assert tool_names(transcript) == ["list_experiments", "wait"]
    assert clock.waits == [30, 30]
    assert foundry.inspected_ids == [experiment_id, experiment_id]
    assert len(sink.notifications) == 1


async def test_monitor_can_yield_when_graph_tool_budget_is_exhausted() -> None:
    """Prove a forced cycle yield cannot loop forever behind the monitor invariant."""
    scenario = scenario_with_lifecycle([ExperimentStatus.IN_QUEUE, ExperimentStatus.DONE])
    experiment_id = scenario.experiment.id
    agent = ScriptedAgent(
        [
            AgentTurn(rationale="Discover.", action=ListExperiments()),
            AgentTurn(
                rationale="Monitor one experiment.",
                action=SelectObjective(
                    kind="monitor",
                    experiment_ids=[experiment_id],
                    summary="Monitor queue progress.",
                ),
            ),
            AgentTurn(
                rationale="The cycle budget is exhausted.",
                action=FinishCycle(summary="Resume monitoring in a fresh cycle."),
            ),
        ]
    )
    foundry, clock = timed_foundry([scenario])
    graph = build_autonomous_graph(
        agent,
        foundry,
        RecordingSink(),
        sleep=clock.sleep,
        clock=clock,
        max_tool_calls_per_cycle=2,
    )

    result = await AgentRuntime(graph).run(max_cycles=1)

    assert result.get("outcome") == "cycle_finished"
    assert result.get("cycle_summary") == "Resume monitoring in a fresh cycle."


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
        first_foundry, first_clock = timed_foundry([first_scenario])
        graph = build_autonomous_graph(
            DemoAutonomousReasoningEngine(),
            first_foundry,
            sink,
            checkpointer=checkpointer,
            sleep=first_clock.sleep,
            clock=first_clock,
        )
        first = await AgentRuntime(graph).run(max_cycles=1)
        restarted_foundry, restarted_clock = timed_foundry(
            [
                first_scenario.model_copy(
                    update={
                        "lifecycle": [
                            LifecycleStep(status=ExperimentStatus.DONE, duration_seconds=0)
                        ]
                    }
                ),
                second_scenario,
            ]
        )
        restarted_graph = build_autonomous_graph(
            DemoAutonomousReasoningEngine(),
            restarted_foundry,
            sink,
            checkpointer=checkpointer,
            sleep=restarted_clock.sleep,
            clock=restarted_clock,
        )
        second = await AgentRuntime(restarted_graph).run(max_cycles=1)

    assert first.get("cycle_count") == 1
    assert second.get("cycle_count") == 2
    assert len(second.get("transcript", [])) == 7
    assert second.get("completed_experiment_ids") == experiment_ids
    assert set(second.get("monitoring", {})) == set(experiment_ids)


@pytest.mark.parametrize("seed", range(10))
async def test_agent_handles_fuzzed_experiment_collections(seed: int) -> None:
    """Fuzz collection order and lifecycle events through the real graph loop."""
    randomizer = random.Random(seed)
    scenarios = load_scenarios()
    randomizer.shuffle(scenarios)
    fuzzed = [
        scenario.model_copy(update={"lifecycle": fuzz_lifecycle(randomizer)})
        for scenario in scenarios
    ]
    expected_ids = [scenario.experiment.id for scenario in fuzzed]
    sink = RecordingSink()
    foundry, clock = timed_foundry(fuzzed)
    graph = build_autonomous_graph(
        DemoAutonomousReasoningEngine(),
        foundry,
        sink,
        sleep=clock.sleep,
        clock=clock,
    )

    result = await AgentRuntime(graph).run(max_cycles=len(fuzzed))

    assert set(result.get("completed_experiment_ids", [])) == set(expected_ids)
    assert len(sink.notifications) == len(fuzzed)
    assert all("results are ready" in notification.title for notification in sink.notifications)
