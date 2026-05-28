"""Composio integration for Hermes.

Bridges Composio actions into the Hermes tool registry:

* Tool handlers return JSON strings via ``tool_result`` / ``tool_error`` to
  match the hermes registry contract.
* User identity is resolved from a ``ContextVar`` set by the plugin's
  ``pre_llm_call`` hook (so each gateway session gets its own Composio
  user), falling back to the ``COMPOSIO_DEFAULT_ENTITY`` env var, then
  to ``"default"``. Composio v0 called this an "entity"; v1 calls it a
  "user". We kept the env-var name for backward compatibility with
  existing deployments.
* Action schemas are cached on disk under ``$HERMES_HOME/composio-cache/``
  so plugin load is fast after the first fetch.

Built against composio v1 (package ``composio``, not the retired
``composio-core`` v0). The v0 connected-accounts endpoint was sunset by
Composio and returns HTTP 410, which is what motivated the v1 migration.
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


_client = None  # composio.Composio v1 client
_composio_available = False
_init_attempted = False

# tool name -> action slug (same string in v1; kept as a set-shaped map for
# the is_composio_tool() membership check below).
_action_map: dict[str, str] = {}

# app name (lowercase) -> list[tool schema dict] (hermes registry shape).
_app_schema_cache: dict[str, list[dict]] = {}

# user_id -> (timestamp, set[str]) connected-app cache.
_connected_apps_cache: dict[str, tuple[float, set[str]]] = {}


# Per-call user override (set by the plugin's pre_llm_call hook from the
# session's sender_id). Thread/task-safe via contextvars — each gateway
# session's chain of tool calls runs in its own context.
_current_entity: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "composio_user_id", default=None,
)


def set_current_entity(entity_id: str | None) -> None:
    """Called by the plugin's pre_llm_call hook once per turn.

    Kept named ``set_current_entity`` for callsite stability; the value is
    the Composio v1 user_id.
    """
    _current_entity.set(entity_id or None)


def current_entity_id() -> str:
    """Resolve the Composio user_id for the current call.

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
    global _client, _composio_available, _init_attempted
    if _init_attempted:
        return
    _init_attempted = True

    api_key = os.environ.get("COMPOSIO_API_KEY")
    if not api_key:
        logger.info("COMPOSIO_API_KEY not set — Composio tools disabled")
        return

    try:
        from composio import Composio  # type: ignore[import-not-found]
        # toolkit_versions="latest" pins every toolkit call to the most recent
        # action contract. Without it, composio 0.13+ raises
        # ToolVersionRequiredError on every tools.execute() call ("Toolkit
        # version not specified."). We're not in a position to pin individual
        # action versions per toolkit — the agent discovers actions
        # dynamically — so the "latest" default matches our discovery model.
        _client = Composio(api_key=api_key, toolkit_versions="latest")
        _composio_available = True
        logger.info("Composio v1 client initialized")
    except ImportError:
        logger.warning(
            "composio package not installed — run `pip install 'hermes-agent[composio]'` "
            "or `pip install composio` to enable external app tools"
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


def _schema_from_tool(tool: Any) -> dict | None:
    """Map a composio v1 ``Tool`` model to a hermes registry schema dict."""
    slug = getattr(tool, "slug", None) or getattr(tool, "name", None)
    if not slug:
        return None
    description = (
        getattr(tool, "human_description", None)
        or getattr(tool, "description", "")
        or ""
    )
    # v1 ``input_parameters`` is a JSON-Schema dict with at least ``type`` and
    # ``properties``; older variants used ``name``/``parameters``. Fall back to
    # an empty object schema rather than dropping the tool.
    params = getattr(tool, "input_parameters", None) or {}
    if not isinstance(params, dict):
        params = {}
    parameters = {
        "type": params.get("type", "object"),
        "properties": params.get("properties", {}) or {},
    }
    required = params.get("required")
    if required:
        parameters["required"] = list(required)
    return {
        "name": slug,
        "description": description,
        "parameters": parameters,
    }


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
        tools = _client.tools.get_raw_composio_tools(toolkits=[key])
    except Exception:
        logger.exception("Failed to fetch Composio schemas for '%s'", key)
        _app_schema_cache[key] = []
        return []

    result: list[dict] = []
    for tool in tools:
        schema = _schema_from_tool(tool)
        if not schema:
            continue
        _action_map[schema["name"]] = schema["name"]
        result.append(schema)

    _app_schema_cache[key] = result
    _write_disk_cache(key, result)
    logger.info("Cached %d composio actions for '%s'", len(result), key)
    return result


def is_composio_tool(tool_name: str) -> bool:
    return tool_name in _action_map


# ----- connections -----------------------------------------------------------


def get_connected_apps(entity_id: str) -> set[str]:
    """Set of app names this user has active connections for (TTL-cached)."""
    _init()
    if not _composio_available:
        return set()
    now = time.time()
    entry = _connected_apps_cache.get(entity_id)
    if entry and (now - entry[0]) < CONNECTIONS_TTL:
        return entry[1]
    try:
        response = _client.connected_accounts.list(
            user_ids=[entity_id],
            statuses=["ACTIVE"],
        )
    except Exception:
        logger.exception("Failed to list Composio connections for %s", entity_id)
        return set()

    apps: set[str] = set()
    for item in getattr(response, "items", None) or []:
        toolkit = getattr(item, "toolkit", None)
        slug = getattr(toolkit, "slug", None) if toolkit else None
        if slug:
            apps.add(slug.lower())
    _connected_apps_cache[entity_id] = (now, apps)
    return apps


def invalidate_connections(entity_id: str) -> None:
    _connected_apps_cache.pop(entity_id, None)


def initiate_connection(entity_id: str, app_name: str) -> tuple[str | None, str | None]:
    """Start OAuth for *app_name* under *entity_id*.

    Returns ``(redirect_url, error)``. On success ``error`` is ``None``;
    on failure ``redirect_url`` is ``None`` and ``error`` carries the
    Composio exception message so the handler can surface a real error
    (instead of the old swallowed-to-None contract that produced the
    misleading "check COMPOSIO_API_KEY" message).
    """
    _init()
    if not _composio_available:
        return None, "Composio is not configured (set COMPOSIO_API_KEY)."
    try:
        # toolkits.authorize() finds or auto-creates a Composio-managed auth
        # config for the toolkit, so callers don't have to provision one in
        # the dashboard before connecting.
        request = _client.toolkits.authorize(
            user_id=str(entity_id),
            toolkit=app_name,
        )
        invalidate_connections(entity_id)
        url = getattr(request, "redirect_url", None)
        if not url:
            return None, (
                f"Composio returned no redirect URL for '{app_name}'. "
                "Check the app slug is valid (gmail, googlecalendar, slack, ...)."
            )
        return url, None
    except Exception as exc:
        logger.exception("Failed to initiate %s connection for %s", app_name, entity_id)
        return None, f"{type(exc).__name__}: {exc}"


def check_connection(entity_id: str, app_name: str) -> bool:
    _init()
    if not _composio_available:
        return False
    try:
        response = _client.connected_accounts.list(
            user_ids=[str(entity_id)],
            toolkit_slugs=[app_name],
            statuses=["ACTIVE"],
        )
    except Exception:
        return False
    return bool(getattr(response, "items", None))


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
        result = _client.tools.execute(
            slug=tool_name,
            arguments=params,
            user_id=str(entity_id),
        )
    except Exception as e:
        logger.exception("Composio tool %s failed", tool_name)
        return False, f"Error executing {tool_name}: {e}"

    # v1 ToolExecutionResponse is a TypedDict subclass of dict with keys
    # ``data``, ``error``, ``successful``.
    if isinstance(result, dict):
        successful = result.get("successful")
        if successful is False:
            err = result.get("error") or result.get("data") or "Unknown error"
            return False, f"Error: {tool_name} failed — {err}"
        if result.get("error"):
            return False, f"Error: {result['error']}"
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
