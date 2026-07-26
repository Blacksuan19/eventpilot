"""Serve typed agent events through a lightweight live dashboard."""

import asyncio
import hashlib
import json
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable
from importlib.resources import files
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from eventpilot.core.approvals import ApprovalDecision
from eventpilot.core.reporting import AgentEvent
from eventpilot.dashboard.supervisor import AgentHealth


class DashboardEventStore:
    """Retain recent agent events and broadcast them to browser subscribers."""

    def __init__(self, *, max_events: int = 5_000, path: Path | None = None) -> None:
        """Create a bounded event history, optionally hydrated from durable JSON Lines."""
        self._events: deque[dict[str, Any]] = deque(maxlen=max_events)
        self._fingerprints: set[str] = set()
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
        self._path = path
        if path and path.exists():
            for line in path.read_text(encoding="utf-8").splitlines()[-max_events:]:
                self._append(json.loads(line))

    def emit(self, event: AgentEvent) -> None:
        """Store and broadcast one complete typed runtime event."""
        payload = event.model_dump(mode="json")
        if not self._append(payload):
            return
        if self._path:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as event_log:
                event_log.write(f"{json.dumps(payload, separators=(',', ':'))}\n")
        for subscriber in self._subscribers:
            subscriber.put_nowait(payload)

    def snapshot(self) -> list[dict[str, Any]]:
        """Return a copy of retained events in chronological order."""
        return list(self._events)

    def clear(self) -> None:
        """Remove retained and durable events for a fresh demonstration run."""
        self._events.clear()
        self._fingerprints.clear()
        if self._path:
            self._path.unlink(missing_ok=True)

    def _append(self, payload: dict[str, Any]) -> bool:
        """Append one semantic event unless durable replay already recorded it."""
        fingerprint = self._fingerprint(payload)
        if fingerprint in self._fingerprints:
            return False
        if len(self._events) == self._events.maxlen:
            self._fingerprints.discard(self._fingerprint(self._events[0]))
        self._events.append(payload)
        self._fingerprints.add(fingerprint)
        return True

    @staticmethod
    def _fingerprint(payload: dict[str, Any]) -> str:
        """Hash event meaning while ignoring its replay-dependent timestamp."""
        semantic_payload = {key: value for key, value in payload.items() if key != "timestamp"}
        encoded = json.dumps(semantic_payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        """Register and return a queue for one live browser connection."""
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        """Remove a disconnected browser queue."""
        self._subscribers.discard(queue)


ResetAgent = Callable[[], Awaitable[None]]
ResolveApproval = Callable[[str, ApprovalDecision], Awaitable[bool]]
GetAgentHealth = Callable[[], AgentHealth]


class ApprovalResponse(BaseModel):
    """Validate an operator decision submitted by an interactive client."""

    decision: ApprovalDecision


def create_dashboard_app(
    store: DashboardEventStore,
    *,
    reset_agent: ResetAgent | None = None,
    resolve_approval: ResolveApproval | None = None,
    get_agent_health: GetAgentHealth | None = None,
) -> FastAPI:
    """Create the HTTP and server-sent-event interface for one event store."""
    app = FastAPI(title="EventPilot Live", docs_url=None, redoc_url=None)

    @app.get("/", response_class=HTMLResponse)
    async def dashboard() -> str:
        """Return the self-contained presentation dashboard."""
        return files("eventpilot.dashboard").joinpath("index.html").read_text(encoding="utf-8")

    @app.get("/api/events")
    async def events() -> dict[str, Any]:
        """Return retained events for initial page hydration."""
        return {"events": store.snapshot()}

    @app.get("/api/health")
    async def health() -> JSONResponse:
        """Report both HTTP readiness and the supervised agent lifecycle."""
        agent = get_agent_health() if get_agent_health else AgentHealth("stopped")
        healthy = agent.state in {"starting", "running"}
        return JSONResponse(
            {
                "status": "ok" if healthy else "degraded",
                "agent": {"state": agent.state, "detail": agent.detail},
            },
            status_code=200 if healthy else 503,
        )

    @app.post("/api/reset")
    async def reset() -> dict[str, str]:
        """Reset the durable agent runtime and dashboard history when configured."""
        if reset_agent is None:
            return {"status": "unavailable"}
        await reset_agent()
        return {"status": "reset"}

    @app.post("/api/approvals/{approval_id}")
    async def resolve(approval_id: str, response: ApprovalResponse) -> dict[str, str]:
        """Resume a suspended tool call with an operator's decision."""
        if resolve_approval is None or not await resolve_approval(approval_id, response.decision):
            raise HTTPException(status_code=404, detail="Pending approval not found")
        return {"status": response.decision.value}

    @app.get("/api/stream")
    async def stream() -> StreamingResponse:
        """Stream new agent events to the dashboard over SSE."""
        queue = store.subscribe()

        async def event_stream() -> AsyncIterator[str]:
            try:
                while True:
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=15)
                        yield f"data: {json.dumps(event, separators=(',', ':'))}\n\n"
                    except TimeoutError:
                        yield ": heartbeat\n\n"
            finally:
                store.unsubscribe(queue)

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return app
