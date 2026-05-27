"""Composio plugin — connect Hermes to external apps via Composio.

Wiring:

1. **Lifecycle tools** (always registered when the plugin loads):
   ``composio_connect``, ``composio_list_connections``,
   ``composio_check_connection``, ``composio_disconnect_app``.
2. **Action tools**: every action of each app in ``COMPOSIO_APPS`` is
   registered at plugin load into the ``composio`` toolset. Hermes's
   agent resolves ``self.tools`` once per agent init, so we eagerly
   register — lazy-loading mid-turn wouldn't reach the LLM on this
   codebase (see notes in README).
3. **pre_llm_call hook** — sets the per-turn Composio entity id from the
   session's ``sender_id`` and appends an ``<external_apps>`` block to
   the current user message listing what's connected.
"""

from __future__ import annotations

import logging
from typing import Any

from tools.registry import registry, tool_error, tool_result

from . import composio_bridge as bridge

logger = logging.getLogger(__name__)


TOOLSET = "composio"


# ---------------------------------------------------------------------------
# Lifecycle tool handlers
# ---------------------------------------------------------------------------


def _check_api_key_available() -> bool:
    return bridge.is_available()


def _handle_connect(args: dict[str, Any], **_: Any) -> str:
    app = (args.get("app") or "").strip().lower()
    if not app:
        return tool_error("Missing required argument: app")

    entity_id = bridge.current_entity_id()

    if bridge.check_connection(entity_id, app):
        return tool_result(
            status="already_connected",
            app=app,
            entity_id=entity_id,
            message=f"{app} is already connected for entity '{entity_id}'.",
        )

    url, err = bridge.initiate_connection(entity_id, app)
    if not url:
        return tool_error(
            f"Failed to start connection for '{app}': {err or 'no redirect URL returned'}"
        )
    return tool_result(
        status="pending",
        app=app,
        entity_id=entity_id,
        redirect_url=url,
        message=(
            f"Open this URL in a browser to authorize {app}. Once authorized, "
            "the connection becomes active immediately (cache TTL 5min)."
        ),
    )


def _handle_list_connections(args: dict[str, Any], **_: Any) -> str:
    entity_id = bridge.current_entity_id()
    apps = sorted(bridge.get_connected_apps(entity_id))
    return tool_result(entity_id=entity_id, connected_apps=apps)


def _handle_check_connection(args: dict[str, Any], **_: Any) -> str:
    app = (args.get("app") or "").strip().lower()
    if not app:
        return tool_error("Missing required argument: app")
    entity_id = bridge.current_entity_id()
    connected = bridge.check_connection(entity_id, app)
    return tool_result(entity_id=entity_id, app=app, connected=connected)


def _handle_disconnect_app(args: dict[str, Any], **_: Any) -> str:
    """Deregister local action tools for an app (does NOT revoke OAuth).

    Use this to trim the per-turn tool budget without logging the user out.
    To truly disconnect, revoke the app from the Composio dashboard.
    """
    app = (args.get("app") or "").strip().lower()
    if not app:
        return tool_error("Missing required argument: app")

    # Pull the cached schemas (no network call) and deregister each.
    schemas = bridge._app_schema_cache.get(app, [])
    removed: list[str] = []
    for schema in schemas:
        name = schema.get("name")
        if name and registry.get_entry(name):
            registry.deregister(name)
            bridge._action_map.pop(name, None)
            removed.append(name)
    bridge._app_schema_cache.pop(app, None)
    return tool_result(
        app=app,
        action_tools_removed=len(removed),
        note=(
            "Local action tools for this app are no longer visible to the LLM. "
            "OAuth is still active — run composio_connect to re-expose the tools, "
            "or revoke from the Composio dashboard to fully disconnect."
        ),
    )


# ---------------------------------------------------------------------------
# Dynamic Composio action handler (one function, closed over action name)
# ---------------------------------------------------------------------------


def _make_action_handler(action_name: str):
    def _handler(args: dict[str, Any], **_: Any) -> str:
        entity_id = bridge.current_entity_id()
        ok, text = bridge.execute(action_name, args or {}, entity_id)
        if not ok:
            return tool_error(text)
        return tool_result(result=text)

    return _handler


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


_CONNECT_SCHEMA = {
    "name": "composio_connect",
    "description": (
        "Start an OAuth connection flow for an external app via Composio. Returns a "
        "redirect URL for the user to authorize. Use this when the user asks to "
        "connect Gmail, Google Calendar, Slack, GitHub, Notion, etc."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "app": {
                "type": "string",
                "description": (
                    "Composio app slug (lowercase). Common values: gmail, "
                    "googlecalendar, slack, github, notion, googledrive, googledocs, "
                    "linkedin."
                ),
            },
        },
        "required": ["app"],
    },
}


_LIST_CONNECTIONS_SCHEMA = {
    "name": "composio_list_connections",
    "description": (
        "List the external apps the current user has connected via Composio."
    ),
    "parameters": {"type": "object", "properties": {}},
}


_CHECK_CONNECTION_SCHEMA = {
    "name": "composio_check_connection",
    "description": (
        "Return whether the current user has an active Composio connection for the "
        "given app."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "app": {
                "type": "string",
                "description": "Composio app slug (e.g. gmail, slack).",
            },
        },
        "required": ["app"],
    },
}


_DISCONNECT_APP_SCHEMA = {
    "name": "composio_disconnect_app",
    "description": (
        "Deregister the action tools for an app from this session's tool list. Does "
        "NOT revoke OAuth — use this only to free tool-budget space."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "app": {
                "type": "string",
                "description": "Composio app slug whose action tools to hide.",
            },
        },
        "required": ["app"],
    },
}


# ---------------------------------------------------------------------------
# pre_llm_call hook
# ---------------------------------------------------------------------------


def _on_pre_llm_call(
    session_id: str = "",
    sender_id: str = "",
    is_first_turn: bool = False,
    **_: Any,
):
    """Set the entity id for this turn and inject a connected-apps summary.

    Hermes appends the returned string to the current user message (see
    ``run_agent.py`` pre_llm_call wiring). Returning an empty dict/string
    is a no-op.
    """
    # Prefer the platform-supplied sender_id; fall back to env/default.
    bridge.set_current_entity(sender_id or None)

    if not bridge.is_available():
        return None

    entity_id = bridge.current_entity_id()
    apps_summary = bridge.describe_connected_apps(entity_id)
    if not apps_summary or apps_summary == "(no external apps connected)":
        return None

    context = (
        "<external_apps>\n"
        "The user has the following external apps connected via Composio. Call their "
        "actions directly (tool names like GMAIL_*, GOOGLECALENDAR_*, SLACK_*) when "
        "relevant. If the user asks to connect a new app, use `composio_connect`.\n\n"
        f"{apps_summary}\n"
        "</external_apps>"
    )
    return {"context": context}


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------


def _register_lifecycle_tools(ctx) -> None:
    ctx.register_tool(
        name="composio_connect",
        toolset=TOOLSET,
        schema=_CONNECT_SCHEMA,
        handler=_handle_connect,
        check_fn=_check_api_key_available,
        requires_env=["COMPOSIO_API_KEY"],
        description=_CONNECT_SCHEMA["description"],
        emoji="🔗",
    )
    ctx.register_tool(
        name="composio_list_connections",
        toolset=TOOLSET,
        schema=_LIST_CONNECTIONS_SCHEMA,
        handler=_handle_list_connections,
        check_fn=_check_api_key_available,
        requires_env=["COMPOSIO_API_KEY"],
        description=_LIST_CONNECTIONS_SCHEMA["description"],
        emoji="🧩",
    )
    ctx.register_tool(
        name="composio_check_connection",
        toolset=TOOLSET,
        schema=_CHECK_CONNECTION_SCHEMA,
        handler=_handle_check_connection,
        check_fn=_check_api_key_available,
        requires_env=["COMPOSIO_API_KEY"],
        description=_CHECK_CONNECTION_SCHEMA["description"],
        emoji="✅",
    )
    ctx.register_tool(
        name="composio_disconnect_app",
        toolset=TOOLSET,
        schema=_DISCONNECT_APP_SCHEMA,
        handler=_handle_disconnect_app,
        check_fn=_check_api_key_available,
        requires_env=["COMPOSIO_API_KEY"],
        description=_DISCONNECT_APP_SCHEMA["description"],
        emoji="🚪",
    )


def _register_app_actions(ctx) -> None:
    """Register each action of each app in COMPOSIO_APPS as a hermes tool.

    Safe to call even when the API key is missing — ``get_app_schemas``
    returns an empty list in that case.
    """
    apps = bridge.configured_apps()
    if not apps:
        return

    total = 0
    for app in apps:
        schemas = bridge.get_app_schemas(app)
        for schema in schemas:
            name = schema.get("name")
            if not name:
                continue
            ctx.register_tool(
                name=name,
                toolset=TOOLSET,
                schema=schema,
                handler=_make_action_handler(name),
                check_fn=_check_api_key_available,
                requires_env=["COMPOSIO_API_KEY"],
                description=schema.get("description", ""),
                emoji="🧩",
            )
            total += 1
        logger.info("Composio: registered %d action(s) for '%s'", len(schemas), app)
    if total:
        logger.info(
            "Composio plugin loaded: %d action tool(s) across %d app(s) in toolset '%s'",
            total, len(apps), TOOLSET,
        )


def register(ctx) -> None:
    _register_lifecycle_tools(ctx)
    _register_app_actions(ctx)
    ctx.register_hook("pre_llm_call", _on_pre_llm_call)
