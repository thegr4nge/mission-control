"""Push mission state from the MCP server to the browser bridge, without slowing a tool call.

Tool handlers call `publish(kind, result, geometry)`. The geometry (Lambert arcs, planet tracks,
about 20 ms) is built on the caller's thread, so every pykep and SPICE call stays on one thread
(CSPICE is not thread-safe); only the finished JSON goes to a daemon thread that sends it to the
bridge websocket. If no bridge is running the event is dropped: the physics tools work the same
with or without a browser open. Queued events are flushed when the process exits.

MISSION_CONTROL_BRIDGE overrides the address (default ws://127.0.0.1:8765/ws); set it to
"off" to disable pushing.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import queue
import threading
import time
from typing import Any, Callable

log = logging.getLogger("mission_control.push")

DEFAULT_URL = "ws://127.0.0.1:8765/ws"
SEND_TIMEOUT_S = 5.0
_QUEUE: "queue.Queue[str]" = queue.Queue(maxsize=200)
_worker: threading.Thread | None = None
_lock = threading.Lock()
_busy = threading.Event()  # set while the worker holds a message it has not finished with


def bridge_url() -> str | None:
    url = os.environ.get("MISSION_CONTROL_BRIDGE", DEFAULT_URL).strip()
    return None if url.lower() in ("", "off", "0", "false") else url


def _enqueue(event: dict[str, Any], what: str) -> None:
    try:
        text = json.dumps({"type": "publish", "event": event}, allow_nan=False)
    except (TypeError, ValueError) as exc:  # NaN or a non-JSON type: a bug upstream; report, don't raise
        log.error("unserialisable %s event: %s", what, exc)
        return
    _ensure_worker()
    try:
        _QUEUE.put_nowait(text)
    except queue.Full:
        log.warning("bridge queue full; dropping %s event", what)


def publish(kind: str, result: dict[str, Any], geometry: Callable[[], dict] | None = None) -> None:
    """Send one tool result (and its 3D geometry) to the browser. Never raises."""
    if bridge_url() is None:
        return
    event = {"type": "mission", "kind": kind, "time_unix": time.time(), "result": result}
    if geometry is not None:
        try:
            event.update(geometry())
        except Exception as exc:  # geometry is a nicety; never lose the result over it
            event["geometry_error"] = f"{type(exc).__name__}: {exc}"
    _enqueue(event, kind)


def publish_chat(role: str, text: str) -> None:
    """A crew member's message, for the browser's Crew tab. Never raises, never blocks."""
    if bridge_url() is None:
        return
    _enqueue({"type": "chat", "time_unix": time.time(), "role": role, "text": text}, "chat")


def _ensure_worker() -> None:
    global _worker
    with _lock:
        if _worker is None or not _worker.is_alive():
            _worker = threading.Thread(target=_run, name="bridge-push", daemon=True)
            _worker.start()


def _take() -> str:
    msg = _QUEUE.get()
    _busy.set()
    return msg


def _run() -> None:
    from websockets.exceptions import WebSocketException
    from websockets.sync.client import connect

    carry: str | None = None  # a message whose send failed; retried once on a fresh connection
    while True:
        msg = carry if carry is not None else _take()
        carry = None
        url = bridge_url()
        if url is None:
            _busy.clear()
            continue
        try:
            with connect(url, open_timeout=1.0, close_timeout=1.0, max_size=2**22) as conn:
                conn.send(json.dumps({"type": "hello", "role": "publisher"}))
                while True:
                    try:
                        conn.send(msg)
                    except (OSError, WebSocketException, TimeoutError):
                        carry = msg
                        break
                    _busy.clear()
                    msg = _take()
                    if bridge_url() != url:
                        carry = msg
                        break
        except (OSError, WebSocketException, TimeoutError):
            # No bridge: drop this message and anything queued meanwhile; try again on the next one.
            while not _QUEUE.empty():
                try:
                    _QUEUE.get_nowait()
                except queue.Empty:
                    break
        if carry is None:
            _busy.clear()


def flush(timeout_s: float = SEND_TIMEOUT_S) -> bool:
    """Wait until every queued message has been sent or dropped."""
    end = time.time() + timeout_s
    while time.time() < end:
        if _QUEUE.empty() and not _busy.is_set():
            return True
        time.sleep(0.02)
    return False


@atexit.register
def _flush_at_exit() -> None:
    if _worker is not None and _worker.is_alive():
        flush()
