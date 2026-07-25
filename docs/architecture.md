# Autonomous agent architecture

EventPilot runs as one long-lived process in a Docker container. Docker starts and recovers that
process; it does not schedule experiment checks. Once started, the LLM controls the work by choosing
tools.

## Runtime boundary

```text
Docker container
└── AgentRuntime (`while True`)
    └── finite LangGraph cycle (stable SQLite supervisor thread)
        └── LLM agent ◄──► tool executor
            ├── list_experiments(limit, offset)
            ├── select_objective(kind, experiment_ids, summary)
            ├── get_experiment(experiment_id)
            ├── list_experiment_updates(experiment_id)
            ├── list_experiment_results(experiment_id)
            ├── send_update(experiment_ids, title, body, priority) ──► END
            ├── wait(seconds, reason) ──► LLM agent
            └── finish_cycle(summary) ──► END
```

The LLM is the recurring control node and selects an Instructor-validated tool call on every turn.
Discovery must precede a typed objective selection. LangGraph stores that objective and rejects API
or update calls outside its experiment scope. Valid calls are appended to the working transcript
before control returns to the LLM.

The graph has explicit `discovery`, `objective`, and `active` phases. Each phase routes only its
available tools. API nodes write typed status, result, update, and observation evidence directly to
state; delivery rules never infer operational truth by rescanning the transcript. The transcript is
retained only for LLM context and audit history.

A `monitor` objective is longitudinal by contract: it cannot finish directly, and active-state
reporting requires at least one executed polling wait. Completed-result, incident, and digest
objectives retain their distinct terminal rules, so waiting is enforced only where it is meaningful.

The stable thread persists `last_checked_at`, `last_observed_status`, `last_reported_status`, and
`next_poll_at` for each monitored experiment. Discovery suppresses experiments before their next
poll window, while delivery schedules the following check using the model's latest wait interval.
An unchanged active status cannot be reported twice.

`AgentRuntime` immediately starts a fresh graph invocation after `send_update` or `finish_cycle`. It
contains no polling policy and manufactures no events. Finite invocations reset LangGraph's step
budget while a stable thread preserves durable counters and summaries in SQLite.

Each cycle also has a finite tool-call budget. If a provider repeatedly selects equivalent reads,
the reasoning engine yields with `finish_cycle` so context cannot grow without bound; the next cycle
starts from fresh API evidence.

## Lifecycle semantics

`wait` means the current objective is still active. It pauses inside the tool executor for the
model-selected interval and then returns a completion result to the LLM. The LLM can refresh scoped
experiments or inspect another scoped endpoint. No external wake event is involved.

`finish_cycle` means the bounded objective is complete. It records a summary and routes to `END`.
The runtime then starts a fresh discovery cycle. It is not a shutdown tool.

`send_update` is also terminal. The graph records evidenced result deliveries, clears the objective,
and ends the cycle without relying on the model to choose a follow-up completion tool.

An operating-system sleep cannot survive container termination. If the container stops, Docker
restarts the process and the supervisor observes current Foundry state again. SQLite preserves graph
checkpoints; it does not attempt to persist an OS timer. An extra read after recovery is safe.

## Foundry boundary

The `FoundryClient` protocol exposes the documented list, detail, update, and result operations. A
fixture-backed mock implements that boundary for the demo and serves many independently progressing
experiments from stored API records. Foundry schemas remain in the adapter and tool results cross
into the agent as validated JSON.

Each fixture lifecycle step has a private duration. The runtime uses monotonic elapsed time, so
repeated reads cannot advance an experiment. Tests inject a manual clock advanced only by executed
`wait` calls, which makes polling behavior deterministic without revealing the schedule to the LLM.

This design makes discovery autonomous without inventing a webhook or internal event schema. New
documented Foundry operations can be added as tools without changing the agent loop.

## Responsibility split

- The LLM chooses objectives, API tools, action tools, and polling cadence.
- LangGraph owns objective validation, scope enforcement, control flow, cycle state, and checkpoints.
- Instructor validates every model-selected tool and its arguments.
- The Foundry mock owns API schemas, fixture state, and lifecycle progression.
- Action providers own delivery side effects and trusted destinations.
- Docker owns process startup and crash recovery only.
