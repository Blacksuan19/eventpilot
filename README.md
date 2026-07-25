# EventPilot

EventPilot is an autonomous, LLM-centered agent for long-running Adaptyv Foundry experiments. It
discovers current experiments through an API-shaped Foundry mock, chooses which work deserves attention,
investigates with additional API tools, controls its own polling cadence, sends operator updates,
and starts a fresh objective when its current work is complete.

The repository is a focused demonstration. It uses Instructor for provider-neutral structured LLM
decisions, LangGraph for the agent/tool loop, and SQLite checkpoints for durable state. It does not
require webhooks or an external scheduler.

## Agent loop

```text
LLM agent ◄──► tool executor
             ├── list_experiments
             ├── get_experiment
             ├── list_experiment_updates
             ├── list_experiment_results
             ├── send_update
             ├── wait
             └── finish_cycle ──► END
```

The LLM selects exactly one validated tool call on every turn. Tool results are appended to its
working transcript and control returns to the LLM. `wait` pauses inside the active graph invocation;
`finish_cycle` ends that finite invocation, and the long-running runtime immediately starts a fresh
discovery cycle on the same SQLite-backed thread.

The Foundry adapter models the documented REST operations directly:

- `GET /api/v1/experiments`
- `GET /api/v1/experiments/{experiment_id}`
- `GET /api/v1/experiments/{experiment_id}/updates`
- `GET /api/v1/experiments/{experiment_id}/results`

The fixture-backed mock implements the `FoundryClient` protocol and serves a normal collection of
experiments. Each experiment advances independently behind that boundary, so the agent discovers
and acts on API state instead of consuming a pre-scripted queue of events.

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

## Quality gate

```bash
make check
```

This verifies the lockfile, linting, formatting, static types, tests with coverage, and both package
artifacts.

## Project shape

```text
src/eventpilot/
├── adapters/       # Foundry API models and protocol
├── core/           # Agent decisions, LangGraph runtime, and action contracts
├── notifications/  # Operator-update providers
├── prompts/        # Versioned agent instructions
├── cli.py          # Continuous and bounded runtime entrypoints
├── fixtures/       # Mock experiment collections and lifecycle events
└── simulator.py    # Fixture-backed Foundry API mock
```

## Safety boundary

The model controls investigation, cadence, and available actions. Deterministic code retains narrow
responsibilities: validating tool calls and API responses, selecting the
trusted notification destination, persisting checkpoints, and enforcing any future approval gates
for consequential write tools.
