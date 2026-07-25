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
            ├── get_experiment(experiment_id)
            ├── list_experiment_updates(experiment_id)
            ├── list_experiment_results(experiment_id)
            ├── send_update(title, body, priority)
            ├── wait(seconds, reason) ──► LLM agent
            └── finish_cycle(summary) ──► END
```

The graph has no mandatory observation node. The LLM is its recurring control node and selects an
Instructor-validated tool call on every turn. LangGraph executes the call, appends the call and
result to the working transcript, then returns control to the LLM.

`AgentRuntime` immediately starts a fresh graph invocation after `finish_cycle`. It contains no
polling policy and manufactures no events. Finite invocations reset LangGraph's step budget while a
stable thread preserves durable counters and summaries in SQLite.

Each cycle also has a finite tool-call budget. If a provider repeatedly selects equivalent reads,
the reasoning engine yields with `finish_cycle` so context cannot grow without bound; the next cycle
starts from fresh API evidence.

## Lifecycle semantics

`wait` means the current objective is still active. It pauses inside the tool executor for the
model-selected interval and then returns a completion result to the LLM. The LLM can refresh the
same experiment, inspect another endpoint, or change focus. No external wake event is involved.

`finish_cycle` means the bounded objective is complete. It records a summary and routes to `END`.
The runtime then starts a fresh discovery cycle. It is not a shutdown tool.

An operating-system sleep cannot survive container termination. If the container stops, Docker
restarts the process and the supervisor observes current Foundry state again. SQLite preserves graph
checkpoints; it does not attempt to persist an OS timer. An extra read after recovery is safe.

## Foundry boundary

The `FoundryClient` protocol exposes the documented list, detail, update, and result operations. The
authenticated HTTP implementation and offline simulator both implement this protocol. Foundry
schemas remain in the adapter and tool results cross into the agent as validated JSON.

This design makes discovery autonomous without inventing a webhook or internal event schema. New
documented Foundry operations can be added as tools without changing the agent loop.

## Responsibility split

- The LLM chooses objectives, API tools, action tools, and polling cadence.
- LangGraph owns control flow, tool routing, cycle state, and checkpoints.
- Instructor validates every model-selected tool and its arguments.
- Foundry adapters own external schemas, authentication, and HTTP errors.
- Action providers own delivery side effects and trusted destinations.
- Docker owns process startup and crash recovery only.
