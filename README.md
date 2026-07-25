# EventPilot

EventPilot is a generic autonomous agent for operating on events and long-running work. It discovers
what is happening, chooses which tools to use, decides when to check again, takes source-defined
actions, and sends useful updates through a configured notification channel. It runs continuously
without requiring webhooks or an external scheduler.

A `DataSource` defines its domain by registering typed tools for its resources, events, and
operations. These tools can inspect state or change it—for example, reading a result, restarting a
failed workflow, cancelling a job, submitting follow-up work, or updating an incident.

This repository demonstrates that architecture against
[Adaptyv Foundry](https://foundry.adaptyvbio.com/). Foundry access is mocked, but its adapter follows
Adaptyv's published API. Fixture timing and lifecycle data stay hidden behind the adapter.

## What it does

- Discovers events and resources through tools supplied by the configured data source.
- Chooses and executes observational or state-changing tools as work evolves.
- Interleaves concurrent objectives instead of blocking on one resource.
- Chooses its own polling interval with a `wait` tool.
- Uses source evidence to decide whether to act, continue investigating, or notify an operator.
- Delivers updates through a replaceable notification sink and records successful delivery.
- Persists progress, avoids duplicate work, and resumes across finite reasoning cycles.
- Enters a long idle wait when the source has no actionable work.

Instructor constrains every LLM response to a registered Pydantic tool model. LangGraph routes the
selected tool and checkpoints agent state in SQLite. Platform details live behind a `DataSource`
protocol, while delivery lives behind a separate `NotificationSink` protocol. A deployment can pair
any source with any sink without changing the agent graph.

## Agent loop

```mermaid
%%{init: {"theme":"base","themeVariables":{"fontFamily":"Inter, ui-sans-serif, system-ui","lineColor":"#64748b","primaryTextColor":"#172033"},"flowchart":{"curve":"basis","nodeSpacing":34,"rankSpacing":46}}}%%
flowchart TB
    subgraph control[Agent control]
        direction LR
        runtime[Continuous runtime] --> agent[LLM selects one typed tool]
        agent --> router{Generic tool router}
    end

    subgraph tools[Tool execution]
        direction LR

        subgraph source_tools[Source tools]
            direction TB
            source[DataSource adapter] --> api[Platform API or API-shaped mock]
            api --> outcome[Persist observations, actions, and evidence]
        end

        subgraph notify[Notify]
            direction TB
            validate{Evidence and scope valid?}
            validate -->|No| rejected[Record rejected action]
            validate -->|Yes| sink[NotificationSink]
            sink --> channel[Telegram, SMS, email, Slack, or another provider]
            channel --> delivered[Record delivery and deduplicate resource]
            delivered --> ready{Another resource ready?}
        end

        subgraph timing[Pause]
            direction TB
            mode{Source idle?}
            mode -->|No| active_wait[Wait until the next check]
            mode -->|Yes| idle_wait[Enter a long idle wait]
        end
    end

    subgraph lifecycle[Lifecycle]
        direction LR
        next_turn[Continue current cycle]
        cycle_end[Finish finite cycle] --> runtime
    end

    router -->|source tool| source
    router -->|send_alert| validate
    router -->|wait| mode
    router -->|finish_cycle| cycle_end

    outcome --> next_turn
    rejected --> next_turn
    ready -->|Yes| next_turn
    ready -->|No| cycle_end
    active_wait --> next_turn
    idle_wait --> next_turn
    next_turn --> agent

    classDef controlNode fill:#ede9fe,stroke:#7c3aed,color:#3b1d72,stroke-width:1.5px;
    classDef decisionNode fill:#fff7d6,stroke:#d99516,color:#704b05,stroke-width:1.5px;
    classDef integrationNode fill:#e8f2ff,stroke:#2f74d0,color:#173f73,stroke-width:1.5px;
    classDef stateNode fill:#e8f8ef,stroke:#24945a,color:#155b38,stroke-width:1.5px;
    classDef actionNode fill:#fff0e7,stroke:#db6b34,color:#783617,stroke-width:1.5px;
    classDef lifecycleNode fill:#f1f5f9,stroke:#64748b,color:#334155,stroke-width:1.5px;
    classDef dangerNode fill:#fff0f2,stroke:#d9475f,color:#831b2d,stroke-width:1.5px;

    class runtime,agent controlNode;
    class router,validate,ready,mode decisionNode;
    class source,api,sink,channel integrationNode;
    class outcome,delivered stateNode;
    class active_wait,idle_wait actionNode;
    class next_turn,cycle_end lifecycleNode;
    class rejected dangerNode;

    style control fill:#faf8ff,stroke:#c4b5fd,stroke-width:1px
    style tools fill:#f8fafc,stroke:#cbd5e1,stroke-width:1px
    style source_tools fill:#f5f9ff,stroke:#bfdbfe,stroke-width:1px
    style notify fill:#fffdf5,stroke:#fde68a,stroke-width:1px
    style timing fill:#fff9f5,stroke:#fed7aa,stroke-width:1px
    style lifecycle fill:#f8fafc,stroke:#cbd5e1,stroke-width:1px
```

After sending a notification, the agent continues its current cycle. It can work through other ready
resources, call another source tool, or wait. Calling `finish_cycle` closes the current cycle, and
the continuous runtime starts the next one.

## Notification sinks

Notification sinks are pluggable delivery integrations. They let a deployment choose where updates
are sent without coupling a monitored platform to a messaging provider. The same data source can be
paired with Telegram, SMS, email, Slack, an incident-management system, or an internal service.

Every sink implements `NotificationSink.send()`, which accepts a destination and a provider-neutral
`Notification` containing a title, body, and priority. It returns a `DeliveryResult` with its channel
and provider message ID so delivery can be recorded durably.

The included `ConsoleNotificationSink` makes the repository runnable without another service. A
production deployment can implement the same small protocol for any delivery provider:

- A Telegram sink can send to a configured chat with the Bot API.
- An SMS sink can deliver through Twilio or another carrier gateway.
- An email or Slack sink can translate the same notification into its native message format.
- An internal sink can open an incident, publish to a queue, or call an operator platform.

When the agent finds an event worth reporting, it creates the notification from the available
evidence and sends it through the configured sink. Provider credentials and destinations remain
outside the LLM's tool arguments.

## Adaptyv API mock

The demo does not require a Foundry account. `MockFoundryClient` implements the same
`FoundryClient` protocol used by the tool adapter and returns Pydantic models shaped around the
official endpoints:

- [List experiments](https://docs.adaptyvbio.com/api-reference/experiments/list-experiments)
- [Get an experiment](https://docs.adaptyvbio.com/api-reference/experiments/get-experiment)
- [List experiment updates](https://docs.adaptyvbio.com/api-reference/experiments/list-experiment-updates)
- [List results for an experiment](https://docs.adaptyvbio.com/api-reference/experiments/list-results-for-an-experiment)

See Adaptyv's [API introduction](https://docs.adaptyvbio.com/api-reference/api-introduction) for
authentication and the production API conventions.

The packaged fixture contains eight independent experiments. Each has lifecycle steps with hidden
durations. Reads do not advance the fixture; only elapsed clock time does. This lets the real agent
loop discover status changes, wait, inspect results, and send alerts without a scripted event queue.

A production integration only needs a `FoundryClient` implementation that performs authenticated
HTTP requests. The source policy and graph do not depend on the mock.

## Run the dashboard

The dashboard is an optional demonstration surface for observing the running agent. It is not part
of the agent loop and is not required by either the source or notification interfaces.

Copy the example configuration and add LLM credentials:

```bash
cp .env.example .env
```

```dotenv
LLM_PROVIDER=openai
LLM_MODEL=your-model
LLM_API_KEY=your-api-key
LLM_API_BASE=
```

Start the container:

```bash
docker compose up --build
```

Open [http://localhost:8000](http://localhost:8000). The dashboard shows the current decision,
rationale, typed arguments, experiment states, wait countdowns, delivered alerts, and the complete
tool timeline. Dashboard events and LangGraph checkpoints persist in the Docker volume.

Use **Reset demo** in the header to cancel the current run, clear its checkpoint and alert history,
and start again. The container stays running.

## Time compression

Docker sets `EVENTPILOT_TIME_ACCELERATION=3600`. One logical second advances the mock experiment
clock by one hour. Active waits are capped at five physical seconds so a full experiment portfolio
can be shown in a short recording. After all experiments are reported, the idle wait runs for its
full duration so the dashboard visibly stops polling.

These settings control that behavior:

```dotenv
EVENTPILOT_TIME_ACCELERATION=3600
EVENTPILOT_MAX_PHYSICAL_WAIT_SECONDS=5
```

The model does not see the acceleration factor or fixture durations.

## Other entry points

Run the continuous agent without the browser UI:

```bash
uv sync
uv run eventpilot run
```

Run one bounded cycle with the deterministic, credential-free reasoning engine:

```bash
EVENTPILOT_MOCK_LLM=true uv run eventpilot demo
```

## Adding integrations

### Data sources

A source plugin provides:

- Pydantic tool models and descriptions exposed to Instructor.
- A client or command adapter that executes those tools.
- Source-owned state and validation for scope, evidence, deduplication, and waiting.

For example, a GitHub Actions source could expose workflow tools backed by `gh` without changing
the LangGraph runtime or dashboard.

### Notification sinks

A sink implements the provider-neutral delivery protocol:

```python
class NotificationSink(Protocol):
    channel_name: str

    async def send(
        self,
        destination: str,
        notification: Notification,
    ) -> DeliveryResult: ...
```

Register the implementation when constructing the autonomous agent. No graph, source adapter, or
prompt changes are required.

## Project layout

```text
src/eventpilot/
├── adapters/       # API models, client protocols, and mocks
├── core/           # Reasoning, graph runtime, clocks, and reporting
├── dashboard/      # Browser UI and durable event feed
├── notifications/  # Alert sinks
├── prompts/        # Source-neutral agent instructions
├── sources/        # Platform tools, state, and policy
└── cli.py           # Runtime entry points
```

## Development

```bash
make check
```

This checks the lockfile, Ruff, formatting, Pyright, tests with coverage, and package builds.
