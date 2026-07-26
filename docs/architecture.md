# Autonomous agent architecture

EventPilot is a source-agnostic autonomous monitoring runtime. Docker starts and recovers one
long-lived process. A configured data-source plugin supplies platform tools and normalized resource
observations; the LLM decides when and how to use them within graph-enforced monitoring policy.

## Runtime boundary

```text
Docker container
└── AgentRuntime (`while True`)
    └── finite LangGraph invocation (stable SQLite supervisor thread)
        └── LLM agent ◄──► generic tool router
            ├── parallel-safe source reads ──► Send fan-out ──► DataSource ──► reducer
            ├── state-changing source tool ──► configured DataSource
            ├── send_alert(...) ──► configured NotificationSink ──► END
            └── wait(seconds, reason) ──► checkpoint deadline ──► sleep ──► END
```

The core graph knows nothing about experiments, workflow runs, or platform APIs. It owns a generic
resource portfolio containing discovery phase, objective, observations, polling schedule, pending
delivery, and completed resource IDs. Platform tools translate API responses into normalized
`SourceEffect` values that the graph reduces into this durable state.

Instructor builds its response model and agent-visible catalog directly from Pydantic schemas
published by the data source and core. Names, arguments, constraints, and descriptions are not
duplicated in prompts. Adding or replacing a data source therefore changes the LLM's validated tool
set without adding graph nodes or routing branches.

Each source tool declares whether it is safe to execute concurrently. One reasoning turn may select
multiple parallel-safe tools targeting different resources. LangGraph fans those calls out with
`Send`, waits for the superstep, and then reduces their normalized effects in the original action
order. Control tools, writes, approvals, and overlapping resource calls remain single-action turns.

## Data-source contract

A platform integration defines typed API models, a client protocol, tool schemas, and source
instructions. The data source invokes the injected client and translates each response into generic
monitoring effects. The data source provides:

- A stable name and platform-specific instructions.
- A discovery tool and registered tool models with unique `tool` discriminators.
- Declarative evidence prerequisites for tools with lifecycle constraints.
- Parsing and execution for every registered tool.
- Normalized discovery and observation effects returned by tool execution.
- Approval metadata for consequential operations.

Tool handlers receive a `SourceContext` containing graph-owned monitoring state, the current-invocation
transcript, and the clock. They return the platform JSON result plus immutable `SourceEffect`
records. The graph reducer owns scope, portfolio rotation, evidence persistence, deduplication,
polling, alert readiness, and invocation transitions.

`ResourceSnapshot` carries only source-neutral fields: resource identity, current status, whether it
is active, arbitrary evidence, the original payload, and an optional `alert_ready` signal.
`SourceEffect` uses the same signal when a later observation becomes operator-worthy. Result
availability, workflow conclusions, incident severity, and similar platform concepts stay inside
the source-owned evidence mapping.

A GitHub Actions plugin, for example, could register `list_workflow_runs`, `get_workflow_run`,
`list_run_jobs`, and `rerun_workflow` models. Its data source could call `gh` through an injected CLI
client. Its list operation would emit discovered `ResourceSnapshot` values and detail operations
would emit observations. The graph and notification provider would remain unchanged.

## Core tools

`send_alert` is the source-neutral communication action. The graph validates its `resource_ids`
against current scope and normalized evidence before the configured `NotificationSink` receives the
message. After confirmed delivery, the graph records deduplication and monitoring state.

`wait` checkpoints the selected interval and absolute wake deadline before pausing inside the
process. A normal wake records the polling state and reaches the explicit end-invocation node. If
the process exits during the sleep, the runtime resumes LangGraph's unfinished checkpoint with
`None` input and sleeps only for the remaining interval. Once the invocation reaches `END`, the
runtime starts a fresh finite invocation on the same SQLite thread. Runtime policy bounds the
interval to `EVENTPILOT_MAX_WAIT_SECONDS`, which defaults to one hour. The same ceiling appears in
the generated tool schema, while the durable result retains both requested and applied durations.

## Runtime reporting

The graph emits typed `AgentDecisionEvent`, `ToolResultEvent`, and `InvocationFinishedEvent` records
through an `AgentReporter` protocol. Decision events include every concrete Pydantic action model,
validated arguments, whether the turn is parallel, rationale, currently available tools, counters,
and the source-state snapshot.
Tool events add the result, outcome, and resulting state; wait events explicitly expose requested
and elapsed seconds.

The console reporter writes compact JSON lines suitable for Docker logs. The dashboard reporter
retains the complete records and broadcasts them over server-sent events to a read-only browser UI
with current activity, wait progress, source state, alerts, and an expandable timeline. Both
reporters observe the same execution through a composite reporter; neither controls the graph.

The console view summarizes large result collections and state maps while custom reporters receive
the complete typed events and can forward them to tracing, metrics, or audit storage without
changing the graph.

## Adaptyv plugin

The Adaptyv adapter package defines API-shaped Pydantic models, the `FoundryClient` protocol, typed
tool calls, and platform instructions. `AdaptyvDataSource` registers those tools, calls the injected
client, and translates Foundry responses into generic resource snapshots and observations. It
retains only Foundry-specific behavior:

- API response normalization.
- Editable and submittable experiment preconditions.
- Quote inspection and approval copy.
- The demonstration replicate policy.

The fixture-backed `MockFoundryClient` implements the same documented API protocol that a live
client would implement. Private lifecycle durations use an injected clock, allowing deterministic
and accelerated tests without exposing timing metadata to the LLM.

## Responsibility split

- The LLM chooses source tools, alerts, and polling cadence within the configured wait ceiling.
- LangGraph owns generic routing, objectives, resource state, polling, delivery policy, invocation state,
  and checkpoints.
- Instructor validates choices against the dynamically composed tool schema.
- A data source owns platform tools, API translation, and platform-specific preconditions.
- A notification sink owns trusted alert delivery.
- Docker owns process startup and crash recovery only.
