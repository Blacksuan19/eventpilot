# Autonomous agent architecture

EventPilot is a source-agnostic autonomous monitoring runtime. Docker starts and recovers one
long-lived process. A configured data-source plugin supplies platform tools and normalized resource
observations; the LLM decides when and how to use them within graph-enforced monitoring policy.

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

The core graph knows nothing about experiments, workflow runs, or platform APIs. It owns a generic
resource portfolio containing discovery phase, objective, observations, polling schedule, pending
delivery, and completed resource IDs. Platform tools translate API responses into normalized
`SourceEffect` values that the graph reduces into this durable state.

Instructor builds its response model and agent-visible catalog directly from Pydantic schemas
published by the adapter, source, and core. Names, arguments, constraints, and descriptions are not
duplicated in prompts. Adding or replacing an adapter therefore changes the LLM's validated tool
set without adding graph nodes or routing branches.

## Data-source contract

A platform adapter publishes typed API operations and instructions, then executes those operations
through its client transport. The data source provides:

- A stable name and the adapter-provided instructions.
- A discovery tool and source-owned tool models with unique `tool` discriminators.
- Declarative evidence prerequisites for tools with lifecycle constraints.
- Parsing and execution for every registered tool.
- Normalized discovery and observation effects returned by tool execution.
- Approval metadata for consequential operations.

Tool handlers receive a `SourceContext` containing graph-owned monitoring state, the current-cycle
transcript, and the clock. They return the platform JSON result plus immutable `SourceEffect`
records. The graph reducer owns scope, portfolio rotation, evidence persistence, deduplication,
polling, alert readiness, and cycle transitions.

A GitHub Actions plugin, for example, could register `list_workflow_runs`, `get_workflow_run`,
`list_run_jobs`, and `rerun_workflow` models. Its handlers could call `gh` through an injected CLI
adapter. Its list operation would emit discovered `ResourceSnapshot` values and detail operations
would emit observations. The graph and notification provider would remain unchanged.

## Core tools

`send_alert` is the source-neutral communication action. The graph validates its `resource_ids`
against current scope and normalized evidence before the configured `NotificationSink` receives the
message. After confirmed delivery, the graph records deduplication and monitoring state.

`wait` pauses inside the process for the LLM-selected interval. Afterward, the graph records the
requested duration and wake timestamp in its polling state. An operating-system
sleep cannot survive container termination; after Docker restarts the process, the supervisor
observes current platform state again.

`finish_cycle` records a summary and routes to `END` after graph policy approves it. The runtime
immediately starts a fresh finite invocation on the same SQLite thread. It is not a shutdown tool.

## Runtime reporting

The graph emits typed `AgentDecisionEvent`, `ToolResultEvent`, and `CycleFinishedEvent` records
through an `AgentReporter` protocol. Decision events include the concrete Pydantic action model,
validated arguments, rationale, currently available tools, counters, and the source-state snapshot.
Tool events add the result, outcome, and resulting state; wait events explicitly expose requested
and elapsed seconds.

The console reporter writes compact JSON lines suitable for Docker logs. The dashboard reporter
retains the complete records and broadcasts them over server-sent events to a read-only browser UI
with current activity, wait progress, source state, alerts, and an expandable timeline. Both
reporters observe the same execution through a composite reporter; neither controls the graph.

The console view summarizes large result
collections and state maps while custom reporters receive the complete typed events and can forward
them to tracing, metrics, or audit storage without changing the graph.

## Adaptyv plugin

`FoundryToolAdapter` publishes and executes Foundry list, detail, update, and result operations.
`AdaptyvDataSource` executes those operations and translates Foundry responses into generic
resource snapshots and observations. It retains only Foundry-specific behavior:

- API response normalization.
- Editable and submittable experiment preconditions.
- Quote inspection and approval copy.
- The demonstration replicate policy.

The fixture-backed `MockFoundryClient` implements the same documented API protocol that a live
client would implement. Private lifecycle durations use an injected clock, allowing deterministic
and accelerated tests without exposing timing metadata to the LLM.

## Responsibility split

- The LLM chooses source tools, alerts, and polling cadence.
- LangGraph owns generic routing, objectives, resource state, polling, delivery policy, cycle state,
  and checkpoints.
- Instructor validates choices against the dynamically composed tool schema.
- A data source owns platform tools, API translation, and platform-specific preconditions.
- A notification sink owns trusted alert delivery.
- Docker owns process startup and crash recovery only.
