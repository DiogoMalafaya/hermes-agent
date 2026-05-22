"""Composio integration for Hermes.

Bridges Composio actions into the Hermes tool registry:

* Tool handlers return JSON strings via ``tool_result`` / ``tool_error`` to
  match the hermes registry contract.
* Entity identity is resolved from a ``ContextVar`` set by the plugin's
  ``pre_llm_call`` hook (so each gateway session gets its own Composio
  entity), falling back to the ``COMPOSIO_DEFAULT_ENTITY`` env var, then
  to ``"default"``.
* Action schemas are cached on disk under ``$HERMES_HOME/composio-cache/``
  so plugin load is fast after the first fetch.
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)


CONNECTIONS_TTL = 300  # seconds; Composio connection list is slow
MAX_RESULT_CHARS = 8000  # Cap on a single tool result (LLM context defense)
SCHEMA_CACHE_TTL_SECONDS = 7 * 24 * 60 * 60  # 7d — action schemas rarely change


_GMAIL_HEAVY = frozenset({
    "payload", "body", "raw", "internalDate", "sizeEstimate",
    "historyId", "labelIds",
})
_CALENDAR_HEAVY = frozenset({
    "htmlLink", "etag", "iCalUID", "sequence", "reminders", "creator",
    "organizer", "eventType", "kind", "conferenceData", "hangoutLink",
    "created", "updated",
})


_APP_DESCRIPTIONS = {
    "gmail": "Gmail — read, search, send, draft, label emails",
    "googlecalendar": "Google Calendar — list, create, update, delete events",
    "slack": "Slack — channels, messages, DMs",
    "github": "GitHub — repos, issues, PRs, code search",
    "notion": "Notion — pages, databases, search",
    "googledrive": "Google Drive — files, folders, share",
    "googledocs": "Google Docs — create, read, update documents",
    "linkedin": "LinkedIn — profile, posts, search",
}


# ----- module state ----------------------------------------------------------


_toolset = None
_composio_available = False
_init_attempted = False

# tool name -> opaque Composio action reference. Populated as app schemas load.
_action_map: dict[str, str] = {}

# app name (lowercase) -> list[tool schema dict] (hermes registry shape).
_app_schema_cache: dict[str, list[dict]] = {}

# entity_id -> (timestamp, set[str]) connected-app cache.
_connected_apps_cache: dict[str, tuple[float, set[str]]] = {}


# Per-call entity override (set by the plugin's pre_llm_call hook from the
# session's sender_id). Thread/task-safe via contextvars — each gateway
# session's chain of tool calls runs in its own context.
_current_entity: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "composio_entity_id", default=None,
)


def set_current_entity(entity_id: str | None) -> None:
    """Called by the plugin's pre_llm_call hook once per turn."""
    _current_entity.set(entity_id or None)


def current_entity_id() -> str:
    """Resolve the Composio entity id for the current call.

    Priority: ContextVar (set per-turn by pre_llm_call) → env var → "default".
    """
    override = _current_entity.get()
    if override:
        return override
    env = (os.environ.get("COMPOSIO_DEFAULT_ENTITY") or "").strip()
    return env or "default"


# ----- init ------------------------------------------------------------------


def is_available() -> bool:
    _init()
    return _composio_available


def _init() -> None:
    global _toolset, _composio_available, _init_attempted
    if _init_attempted:
        return
    _init_attempted = True

    api_key = os.environ.get("COMPOSIO_API_KEY")
    if not api_key:
        logger.info("COMPOSIO_API_KEY not set — Composio tools disabled")
        return

    try:
        from composio import ComposioToolSet  # type: ignore[import-not-found]
        _toolset = ComposioToolSet(api_key=api_key)
        _composio_available = True
        logger.info("Composio initialized")
    except ImportError:
        logger.warning(
            "composio-core not installed — run `pip install 'hermes-agent[composio]'` "
            "or `pip install composio-core` to enable external app tools"
        )
    except Exception:
        logger.exception("Failed to initialize Composio")


# ----- result trimming -------------------------------------------------------


def _strip_heavy(obj: Any, heavy: frozenset[str]) -> Any:
    if isinstance(obj, dict):
        return {k: _strip_heavy(v, heavy) for k, v in obj.items() if k not in heavy}
    if isinstance(obj, list):
        return [_strip_heavy(x, heavy) for x in obj]
    return obj


def _unwrap(payload: Any) -> Any:
    """Peel Composio's ``response_data`` envelope if it's the only key."""
    while (
        isinstance(payload, dict)
        and len(payload) == 1
        and "response_data" in payload
    ):
        payload = payload["response_data"]
    return payload


def _serialize_payload(tool_name: str, data: Any) -> str:
    heavy: frozenset[str] = frozenset()
    if tool_name.startswith("GMAIL_"):
        heavy = _GMAIL_HEAVY
    elif tool_name.startswith("GOOGLECALENDAR_"):
        heavy = _CALENDAR_HEAVY
    trimmed = _strip_heavy(data, heavy) if heavy else data
    try:
        text = json.dumps(trimmed, ensure_ascii=False, indent=2, default=str)
    except Exception:
        text = str(trimmed)
    if len(text) > MAX_RESULT_CHARS:
        return (
            text[:MAX_RESULT_CHARS]
            + f"\n... (truncated; full result was {len(text)} chars — "
            "call the tool again with a narrower query if you need more)"
        )
    return text


# ----- schema cache (disk) ---------------------------------------------------


def _schema_cache_dir() -> Path:
    try:
        from hermes_constants import get_hermes_home
        base = get_hermes_home()
    except Exception:
        base = Path.home() / ".hermes"
    d = base / "composio-cache"
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return d


def _read_disk_cache(app: str) -> list[dict] | None:
    path = _schema_cache_dir() / f"{app}.json"
    if not path.exists():
        return None
    try:
        if time.time() - path.stat().st_mtime > SCHEMA_CACHE_TTL_SECONDS:
            return None
        raw = json.loads(path.read_text())
        if isinstance(raw, list):
            return raw
    except (OSError, json.JSONDecodeError):
        pass
    return None


def _write_disk_cache(app: str, schemas: list[dict]) -> None:
    path = _schema_cache_dir() / f"{app}.json"
    try:
        path.write_text(json.dumps(schemas, ensure_ascii=False))
    except OSError:
        logger.debug("Failed to write composio schema cache for %s", app, exc_info=True)


# ----- schemas ---------------------------------------------------------------


def get_app_schemas(app_name: str) -> list[dict]:
    """Return hermes-registry-shaped schemas for every action in *app_name*.

    Each entry is an inner function schema dict
    (``{"name": ..., "description": ..., "parameters": ...}``) — suitable
    for direct use as the ``schema`` arg to ``registry.register()``.
    """
    key = app_name.lower()
    if key in _app_schema_cache:
        return _app_schema_cache[key]

    disk = _read_disk_cache(key)
    if disk is not None:
        _app_schema_cache[key] = disk
        for entry in disk:
            _action_map[entry["name"]] = entry["name"]
        logger.debug("Loaded %d composio actions for '%s' from disk cache", len(disk), key)
        return disk

    _init()
    if not _composio_available:
        return []

    try:
        from composio import App  # type: ignore[import-not-found]
        app = App(key)
        action_models = _toolset.get_action_schemas(
            apps=[app],
            check_connected_accounts=False,
        )
    except Exception:
        logger.exception("Failed to fetch Composio schemas for '%s'", key)
        _app_schema_cache[key] = []
        return []

    result: list[dict] = []
    for action in action_models:
        name = getattr(action, "name", None)
        if not name:
            continue
        params = getattr(action, "parameters", None)
        schema = {
            "name": name,
            "description": getattr(action, "description", "") or "",
            "parameters": {
                "type": getattr(params, "type", "object") if params else "object",
                "properties": getattr(params, "properties", {}) if params else {},
            },
        }
        required = getattr(params, "required", None) if params else None
        if required:
            schema["parameters"]["required"] = list(required)
        _action_map[name] = name
        result.append(schema)

    _app_schema_cache[key] = result
    _write_disk_cache(key, result)
    logger.info("Cached %d composio actions for '%s'", len(result), key)
    return result


def is_composio_tool(tool_name: str) -> bool:
    return tool_name in _action_map


# ----- connections -----------------------------------------------------------


def get_connected_apps(entity_id: str) -> set[str]:
    """Set of app names this entity has active connections for (TTL-cached)."""
    _init()
    if not _composio_available:
        return set()
    now = time.time()
    entry = _connected_apps_cache.get(entity_id)
    if entry and (now - entry[0]) < CONNECTIONS_TTL:
        return entry[1]
    try:
        entity = _toolset.get_entity(id=entity_id)
        try:
            connections = entity.get_connections() or []
        except Exception:
            _connected_apps_cache[entity_id] = (now, set())
            return set()
        apps: set[str] = set()
        for conn in connections:
            name = getattr(conn, "appUniqueId", None) or getattr(conn, "appName", None)
            if name:
                apps.add(name.lower())
        _connected_apps_cache[entity_id] = (now, apps)
        return apps
    except Exception:
        logger.exception("Failed to list Composio connections for %s", entity_id)
        return set()


def invalidate_connections(entity_id: str) -> None:
    _connected_apps_cache.pop(entity_id, None)


def initiate_connection(entity_id: str, app_name: str) -> str | None:
    """Start OAuth for *app_name* under *entity_id*. Returns redirect URL."""
    _init()
    if not _composio_available:
        return None
    try:
        entity = _toolset.get_entity(id=entity_id)
        connection = entity.initiate_connection(app_name=app_name)
        invalidate_connections(entity_id)
        return getattr(connection, "redirectUrl", None)
    except Exception:
        logger.exception("Failed to initiate %s connection for %s", app_name, entity_id)
        return None


def check_connection(entity_id: str, app_name: str) -> bool:
    _init()
    if not _composio_available:
        return False
    try:
        entity = _toolset.get_entity(id=entity_id)
        entity.get_connection(app=app_name)
        return True
    except Exception:
        return False


# ----- execution -------------------------------------------------------------


def execute(tool_name: str, args: dict, entity_id: str) -> tuple[bool, str]:
    """Execute a Composio action.

    Returns ``(ok, text)`` where *text* is an LLM-ready string. ``ok`` is
    ``False`` whenever the action could not be performed — misconfiguration,
    transport error, or a Composio/Google API failure (e.g. a rejected
    calendar insert). Callers must surface a real tool error in that case
    rather than reporting success with the error buried in the body.
    """
    _init()
    if not _composio_available:
        return False, "Error: Composio is not configured (set COMPOSIO_API_KEY)."

    params = {k: v for k, v in (args or {}).items() if not k.startswith("_")}
    try:
        result = _toolset.execute_action(
            action=tool_name,
            params=params,
            entity_id=str(entity_id),
        )
    except Exception as e:
        logger.exception("Composio tool %s failed", tool_name)
        return False, f"Error executing {tool_name}: {e}"

    if isinstance(result, dict):
        if result.get("error"):
            return False, f"Error: {result['error']}"
        if result.get("successfull") is False or result.get("successful") is False:
            err = result.get("error") or result.get("data") or "Unknown error"
            return False, f"Error: {tool_name} failed — {err}"
        payload = result.get("data", result)
    else:
        payload = result

    if isinstance(payload, dict) and payload.get("error"):
        return False, f"Error: {payload['error']}"

    return True, _serialize_payload(tool_name, _unwrap(payload))


# ----- human-readable helpers ------------------------------------------------


def describe_connected_apps(entity_id: str) -> str:
    apps = get_connected_apps(entity_id)
    if not apps:
        return "(no external apps connected)"
    lines = [f"- {app}: {_APP_DESCRIPTIONS.get(app, app)}" for app in sorted(apps)]
    return "\n".join(lines)


def configured_apps() -> list[str]:
    """Apps declared via ``COMPOSIO_APPS`` env (comma-separated)."""
    raw = os.environ.get("COMPOSIO_APPS", "")
    return [a.strip().lower() for a in raw.split(",") if a.strip()]


def split_csv_env(name: str) -> Iterable[str]:
    raw = os.environ.get(name, "")
    return (a.strip() for a in raw.split(",") if a.strip())
