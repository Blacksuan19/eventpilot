# Autonomous agent architecture

EventPilot is a source-agnostic autonomous monitoring runtime. Docker starts and recovers one
long-lived process. A configured data-source plugin supplies the platform tools and deterministic
policy; the LLM decides when and how to use them.

## Runtime boundary

```text
Docker container
└── AgentRuntime (`while True`)
    └── finite LangGraph cycle (stable SQLite supervisor thread)
        └── LLM agent ◄──► generic tool router
            ├── data-source tool ──► configured DataSource
            ├── send_alert(...) ──► configured NotificationSink ──► END
            ├── wait(seconds, reason) ──► LLM agent
            └── finish_cycle(summary) ──► END
```

The core graph knows nothing about experiments, workflow runs, statuses, results, or platform APIs.
It persists an opaque `source_state`, dynamically routes the source's registered Pydantic tool
models, and owns only reasoning, alert delivery, waits, cycle completion, and checkpoints.

Instructor builds its response model and agent-visible catalog directly from Pydantic schemas
published by the adapter, source, and core. Names, arguments, constraints, and descriptions are not
duplicated in prompts. Adding or replacing an adapter therefore changes the LLM's validated tool
set without adding graph nodes or routing branches.

## Data-source contract

A platform adapter publishes typed API operations and instructions, then executes those operations
through its client transport. The data source composes those operations with any policy-only tools
and provides:

- A stable name and the adapter-provided instructions.
- Any source-owned tool models with unique `tool` discriminators.
- Deterministic tool availability based on its private durable state.
- Parsing and execution for every registered tool.
- Wait, alert-validation, delivery-recording, and cycle-finish hooks.

Tool handlers receive a `SourceContext` containing the source's state, current-cycle transcript,
clock, and tool budget. They return a JSON result plus the next source state. This lets a plugin
enforce platform-specific ordering, scope, evidence, deduplication, and polling without leaking
those rules into the autonomous runtime.

A GitHub Actions plugin, for example, could register `list_workflow_runs`, `get_workflow_run`,
`list_run_jobs`, and `rerun_workflow` models. Its handlers could call `gh` through an injected CLI
adapter, while its state tracks inspected and alerted run IDs. The graph and notification provider
would remain unchanged.

## Core tools

`send_alert` is the source-neutral communication action. The selected source validates its
`resource_ids` against current scope and evidence before the configured `NotificationSink` receives
the message. After confirmed delivery, the source records any platform-specific deduplication and
monitoring state.

`wait` pauses inside the process for the LLM-selected interval. Afterward, the source receives the
requested duration and wake timestamp so it can update its own polling state. An operating-system
sleep cannot survive container termination; after Docker restarts the process, the supervisor
observes current platform state again.

`finish_cycle` records a summary and routes to `END` after source policy approves it. The runtime
immediately starts a fresh finite invocation on the same SQLite thread. It is not a shutdown tool.

## Adaptyv plugin

`FoundryToolAdapter` publishes and executes Foundry list, detail, update, and result operations.
`AdaptyvDataSource` composes them with its objective-selection policy tool and owns the remaining
experiment-specific behavior:

- Discovery and objective phases.
- Experiment scope validation.
- Status and result evidence.
- Completed-result suppression.
- Poll scheduling and unchanged-status deduplication.
- Alert readiness and delivery recording.

The fixture-backed `MockFoundryClient` implements the same documented API protocol that a live
client would implement. Private lifecycle durations use an injected clock, allowing deterministic
and accelerated tests without exposing timing metadata to the LLM.

## Responsibility split

- The LLM chooses source tools, alerts, and polling cadence.
- LangGraph owns generic routing, control flow, cycle state, and checkpoints.
- Instructor validates choices against the dynamically composed tool schema.
- A data source owns platform tools, API translation, state, and deterministic policy.
- A notification sink owns trusted alert delivery.
- Docker owns process startup and crash recovery only.
