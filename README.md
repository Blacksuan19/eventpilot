# EventPilot

EventPilot is an autonomous, LLM-centered monitoring agent with pluggable data sources and alert
sinks. A source supplies platform-specific typed tools and deterministic policy; the generic
LangGraph runtime chooses work, controls polling, performs source actions, and sends evidenced
alerts. This demo configures an Adaptyv Foundry source backed by an API-shaped mock.

The repository is a focused demonstration. It uses Instructor for provider-neutral structured LLM
decisions, LangGraph for the agent/tool loop, and SQLite checkpoints for durable state. It does not
require webhooks or an external scheduler.

## Agent loop

```text
LLM agent ◄──► generic tool router
             ├── registered source tool ──► DataSource
             ├── send_alert ──► NotificationSink ──► END
             ├── wait
             └── finish_cycle ──► END
```

The LLM selects exactly one validated tool call on every turn. Instructor composes its response
schema and tool descriptions dynamically from adapter, source-policy, and core Pydantic models;
prompts contain behavior rather than duplicated signatures. Source results are appended to the
working transcript and its private state is persisted opaquely by LangGraph. `wait` pauses inside
the active invocation; `send_alert` or `finish_cycle` ends that finite invocation, and the runtime
immediately starts a fresh cycle on the same SQLite-backed thread.

The source plugin owns tool availability, evidence, scope, deduplication, and scheduling rules. A
GitHub Actions source could therefore expose `gh`-backed workflow tools without changing the graph.
The configured Adaptyv plugin preserves per-experiment monitoring records and rejects out-of-scope
calls or duplicate unchanged reports.

The Foundry adapter models the documented REST operations directly:

- `GET /api/v1/experiments`
- `GET /api/v1/experiments/{experiment_id}`
- `GET /api/v1/experiments/{experiment_id}/updates`
- `GET /api/v1/experiments/{experiment_id}/results`

The fixture-backed mock implements the `FoundryClient` protocol and serves a normal collection of
experiments. Each experiment advances independently behind that boundary, so the agent discovers
and acts on API state instead of consuming a pre-scripted queue of events. Fixture lifecycle steps
have hidden durations driven by a monotonic clock. API reads never advance experiment state, and the
agent sees statuses rather than the configured schedule.

## Configuration

Create the local configuration file:

```bash
cp .env.example .env
```

Required for the LLM runtime:

```dotenv
LLM_PROVIDER=...
LLM_MODEL=...
LLM_API_KEY=...
LLM_API_BASE=...
```

`LLM_PROVIDER` and `LLM_MODEL` form Instructor's `provider/model` identifier. `LLM_API_BASE` is
optional for providers that use their standard endpoint. Foundry is always mocked from the packaged
experiment fixtures. `EVENTPILOT_MOCK_LLM=true` selects the credential-free deterministic agent.

## Run

Install the locked environment and run one bounded cycle:

```bash
uv sync
EVENTPILOT_MOCK_LLM=true uv run eventpilot demo
```

The bounded `demo` command caps physical waits at two seconds while reporting the LLM's requested
interval in the tool result. The continuous runtime honors the requested interval exactly.

Run continuously with the configured LLM and mocked Foundry API:

```bash
uv run eventpilot run
```

Or run the same process with durable SQLite storage in Docker:

```bash
docker compose up --build
```

Docker only starts and recovers the process. The agent's `wait` tool controls polling; Docker does
not wake or schedule the graph.

For time-compressed Docker integration runs, set `EVENTPILOT_TIME_ACCELERATION` to a multiplier
greater than `1`. The mock API and graph then share an accelerated hidden clock, while each wait
only yields briefly in real time. The agent still chooses every polling interval without seeing the
fixture durations or acceleration factor.

## Quality gate

```bash
make check
```

This verifies the lockfile, linting, formatting, static types, tests with coverage, and both package
artifacts.

## Project shape

```text
src/eventpilot/
├── adapters/       # External API models, clients, and scoped test doubles
├── core/           # Source-agnostic reasoning and LangGraph runtime
├── sources/        # Pluggable tools, state, and platform policy
├── notifications/  # Alert-delivery providers
├── prompts/        # Generic and source-specific instructions
├── cli.py          # Continuous and bounded runtime entrypoints
└── fixtures/       # Mock experiment collections and lifecycle events
```

## Safety boundary

The model controls investigation, cadence, and registered actions. The generic graph controls tool
routing, alert delivery, cycle termination, and checkpoints. Each source validates its own scope,
evidence, API responses, and future approval gates for consequential platform actions.
