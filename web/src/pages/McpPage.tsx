import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from "react";
import {
  CheckCircle2,
  ExternalLink,
  Package,
  Plug,
  Power,
  RotateCw,
  Server,
  ShieldCheck,
  ShieldOff,
  Trash2,
  X,
  XCircle,
  Zap,
} from "lucide-react";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Button } from "@nous-research/ui/ui/components/button";
import { Select, SelectOption } from "@nous-research/ui/ui/components/select";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { H2 } from "@nous-research/ui/ui/components/typography/h2";
import { api } from "@/lib/api";
import type {
  McpCatalogDiagnostic,
  McpCatalogEntry,
  McpOAuthFlowState,
  McpOAuthStatus,
  McpServer,
  McpServerCreate,
  McpTestResult,
} from "@/lib/api";
import { DeleteConfirmDialog } from "@/components/DeleteConfirmDialog";
import { useToast } from "@nous-research/ui/hooks/use-toast";
import { useConfirmDelete } from "@nous-research/ui/hooks/use-confirm-delete";
import { useModalBehavior } from "@/hooks/useModalBehavior";
import { Toast } from "@nous-research/ui/ui/components/toast";
import { Card, CardContent } from "@nous-research/ui/ui/components/card";
import { Input } from "@nous-research/ui/ui/components/input";
import { Label } from "@nous-research/ui/ui/components/label";
import { usePageHeader } from "@/contexts/usePageHeader";
import { cn, themedBody } from "@/lib/utils";

type Transport = "http" | "stdio";

const OAUTH_POLL_INTERVAL_MS = 1500;

type FlowMap = Record<string, McpOAuthFlowState>;

function isHttpUrl(value: string): boolean {
  return /^https?:\/\//i.test(value.trim());
}

function truncateText(value: string, maxLength: number): string {
  return value.length > maxLength ? value.slice(0, maxLength) + "..." : value;
}

function parseArgs(raw: string): string[] {
  return raw
    .split(/[\s,]+/)
    .map((s) => s.trim())
    .filter(Boolean);
}

function parseEnv(raw: string): Record<string, string> {
  const env: Record<string, string> = {};
  raw
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .forEach((line) => {
      const idx = line.indexOf("=");
      if (idx === -1) return;
      const key = line.slice(0, idx).trim();
      const value = line.slice(idx + 1).trim();
      if (key) env[key] = value;
    });
  return env;
}

const TRANSPORT_TONE: Record<string, "success" | "warning" | "secondary"> = {
  http: "success",
  stdio: "warning",
  unknown: "secondary",
};

function oauthStatusBadge(status: McpOAuthStatus) {
  switch (status) {
    case "starting":
      return (
        <Badge tone="outline" className="text-amber-500 border-amber-500/40">
          Starting…
        </Badge>
      );
    case "url_ready":
      return (
        <Badge tone="outline" className="text-blue-500 border-blue-500/40">
          Awaiting authorization
        </Badge>
      );
    case "completed":
      return (
        <Badge tone="outline" className="text-emerald-500 border-emerald-500/40">
          Connected
        </Badge>
      );
    case "failed":
      return (
        <Badge tone="outline" className="text-rose-500 border-rose-500/40">
          Failed
        </Badge>
      );
    default:
      return null;
  }
}

export default function McpPage() {
  const [servers, setServers] = useState<McpServer[]>([]);
  const [catalog, setCatalog] = useState<McpCatalogEntry[]>([]);
  const [diagnostics, setDiagnostics] = useState<McpCatalogDiagnostic[]>([]);
  const [loading, setLoading] = useState(true);
  const { toast, showToast } = useToast();
  const { setEnd } = usePageHeader();

  // Avoid stale closures inside the long-lived OAuth poll loop.
  const showToastRef = useRef(showToast);
  useEffect(() => {
    showToastRef.current = showToast;
  }, [showToast]);

  // Add server modal state
  const [createModalOpen, setCreateModalOpen] = useState(false);
  const [name, setName] = useState("");
  const [transport, setTransport] = useState<Transport>("http");
  const [url, setUrl] = useState("");
  const [command, setCommand] = useState("");
  const [args, setArgs] = useState("");
  const [env, setEnv] = useState("");
  const [creating, setCreating] = useState(false);
  const closeCreateModal = useCallback(() => setCreateModalOpen(false), []);
  const createModalRef = useModalBehavior({
    open: createModalOpen,
    onClose: closeCreateModal,
  });

  // Test results keyed by server name
  const [testing, setTesting] = useState<string | null>(null);
  const [testResults, setTestResults] = useState<
    Record<string, McpTestResult>
  >({});

  // Enable/disable state
  const [togglingName, setTogglingName] = useState<string | null>(null);
  const [restartNote, setRestartNote] = useState<string | null>(null);

  // Browser-driven OAuth flow state (headless hosts, e.g. Railway).
  const [flows, setFlows] = useState<FlowMap>({});
  const [oauthBusy, setOauthBusy] = useState<Record<string, boolean>>({});
  const [restarting, setRestarting] = useState(false);
  const pollingRef = useRef<Set<string>>(new Set());

  // Catalog install modal state
  const [installEntry, setInstallEntry] = useState<McpCatalogEntry | null>(
    null,
  );
  const [installEnv, setInstallEnv] = useState<Record<string, string>>({});
  const [installingName, setInstallingName] = useState<string | null>(null);
  const closeInstallModal = useCallback(() => setInstallEntry(null), []);
  const installModalRef = useModalBehavior({
    open: installEntry !== null,
    onClose: closeInstallModal,
  });

  const loadServers = useCallback(() => {
    return api
      .getMcpServers()
      .then((res) => setServers(res.servers))
      .catch((e) => showToast(`Error: ${e}`, "error"));
  }, [showToast]);

  const loadCatalog = useCallback(() => {
    return api
      .getMcpCatalog()
      .then((res) => {
        setCatalog(res.entries);
        setDiagnostics(res.diagnostics);
      })
      .catch((e) => showToast(`Error: ${e}`, "error"));
  }, [showToast]);

  useEffect(() => {
    Promise.all([loadServers(), loadCatalog()]).finally(() =>
      setLoading(false),
    );
  }, [loadServers, loadCatalog]);

  // ── OAuth flow: poll a server's handshake until it terminates ──────────
  const pollFlow = useCallback(
    async (serverName: string) => {
      if (pollingRef.current.has(serverName)) return;
      pollingRef.current.add(serverName);
      try {
        while (pollingRef.current.has(serverName)) {
          let state: McpOAuthFlowState;
          try {
            state = await api.getMcpOAuthStatus(serverName);
          } catch (err) {
            showToastRef.current(`Status fetch failed: ${err}`, "error");
            return;
          }
          setFlows((prev) => ({ ...prev, [serverName]: state }));
          if (state.status === "completed" || state.status === "failed") {
            if (state.status === "completed") {
              // Tokens were just written to disk; refresh the list so the
              // "tokens cached" badge updates.
              void loadServers();
            }
            return;
          }
          await new Promise((r) => setTimeout(r, OAUTH_POLL_INTERVAL_MS));
        }
      } finally {
        pollingRef.current.delete(serverName);
      }
    },
    [loadServers],
  );

  // Resume polling for any OAuth server left mid-handshake (e.g. the user
  // navigated away and came back).
  useEffect(() => {
    for (const s of servers) {
      if (s.auth !== "oauth") continue;
      api
        .getMcpOAuthStatus(s.name)
        .then((state) => {
          setFlows((prev) => ({ ...prev, [s.name]: state }));
          if (state.status === "starting" || state.status === "url_ready") {
            void pollFlow(s.name);
          }
        })
        .catch(() => {});
    }
    // Depend only on the set of servers, not on flow state.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [servers]);

  // Stop all polling on unmount.
  useEffect(() => {
    const ref = pollingRef;
    return () => {
      ref.current.clear();
    };
  }, []);

  const handleConnect = useCallback(
    async (serverName: string) => {
      setOauthBusy((b) => ({ ...b, [serverName]: true }));
      try {
        const state = await api.startMcpOAuth(serverName);
        setFlows((prev) => ({ ...prev, [serverName]: state }));
        void pollFlow(serverName);
      } catch (err) {
        showToastRef.current(`Failed to start OAuth: ${err}`, "error");
      } finally {
        setOauthBusy((b) => ({ ...b, [serverName]: false }));
      }
    },
    [pollFlow],
  );

  const handleResetOAuth = useCallback(async (serverName: string) => {
    pollingRef.current.delete(serverName);
    try {
      await api.clearMcpOAuth(serverName);
    } catch {
      // ignore — clearing is best-effort
    }
    setFlows((prev) => {
      const next = { ...prev };
      delete next[serverName];
      return next;
    });
  }, []);

  const handleRestartGateway = useCallback(async () => {
    setRestarting(true);
    try {
      await api.restartGateway();
      showToastRef.current(
        "Gateway restarting — give it a few seconds then refresh.",
        "success",
      );
    } catch (err) {
      showToastRef.current(`Restart failed: ${err}`, "error");
    } finally {
      setRestarting(false);
    }
  }, []);

  const handleCreate = async () => {
    if (!name.trim()) {
      showToast("Name required", "error");
      return;
    }
    if (transport === "http" && !url.trim()) {
      showToast("URL required", "error");
      return;
    }
    if (transport === "stdio" && !command.trim()) {
      showToast("Command required", "error");
      return;
    }
    setCreating(true);
    try {
      const body: McpServerCreate = { name: name.trim() };
      if (transport === "http") {
        body.url = url.trim();
      } else {
        body.command = command.trim();
        const argList = parseArgs(args);
        if (argList.length) body.args = argList;
      }
      const envMap = parseEnv(env);
      if (Object.keys(envMap).length) body.env = envMap;

      await api.addMcpServer(body);
      showToast("Add ✓", "success");
      setName("");
      setUrl("");
      setCommand("");
      setArgs("");
      setEnv("");
      setTransport("http");
      setCreateModalOpen(false);
      loadServers();
    } catch (e) {
      showToast(`Failed to add: ${e}`, "error");
    } finally {
      setCreating(false);
    }
  };

  const handleTest = async (server: McpServer) => {
    setTesting(server.name);
    try {
      const result = await api.testMcpServer(server.name);
      setTestResults((prev) => ({ ...prev, [server.name]: result }));
      if (result.ok) {
        showToast(`${server.name}: ${result.tools.length} tool(s)`, "success");
      } else {
        showToast(`${server.name}: ${result.error ?? "Failed"}`, "error");
      }
    } catch (e) {
      showToast(`Error: ${e}`, "error");
    } finally {
      setTesting(null);
    }
  };

  const handleToggleEnabled = async (server: McpServer) => {
    const next = !server.enabled;
    setTogglingName(server.name);
    try {
      await api.setMcpServerEnabled(server.name, next);
      setServers((prev) =>
        prev.map((s) =>
          s.name === server.name ? { ...s, enabled: next } : s,
        ),
      );
      setRestartNote(
        "Enable/disable takes effect on the next gateway restart.",
      );
    } catch (e) {
      showToast(`Error: ${e}`, "error");
    } finally {
      setTogglingName(null);
    }
  };

  const serverDelete = useConfirmDelete({
    onDelete: useCallback(
      async (serverName: string) => {
        try {
          await api.removeMcpServer(serverName);
          showToast(`Delete: "${truncateText(serverName, 30)}"`, "success");
          setTestResults((prev) => {
            const next = { ...prev };
            delete next[serverName];
            return next;
          });
          loadServers();
        } catch (e) {
          showToast(`Error: ${e}`, "error");
          throw e;
        }
      },
      [loadServers, showToast],
    ),
  });

  // ── Catalog install ──────────────────────────────────────────────────
  const runInstall = useCallback(
    async (entry: McpCatalogEntry, envMap: Record<string, string>) => {
      setInstallingName(entry.name);
      try {
        const res = await api.installMcpCatalogEntry(entry.name, envMap, true);
        if (res.background) {
          showToast("Installing in background…", "success");
        } else {
          showToast(`Installed: "${truncateText(entry.name, 30)}"`, "success");
        }
        setInstallEntry(null);
        setInstallEnv({});
        await Promise.all([loadServers(), loadCatalog()]);
      } catch (e) {
        showToast(`Failed to install: ${e}`, "error");
      } finally {
        setInstallingName(null);
      }
    },
    [loadServers, loadCatalog, showToast],
  );

  const handleInstallClick = (entry: McpCatalogEntry) => {
    if (entry.required_env.length > 0) {
      const initial: Record<string, string> = {};
      entry.required_env.forEach((item) => {
        initial[item.name] = "";
      });
      setInstallEnv(initial);
      setInstallEntry(entry);
    } else {
      void runInstall(entry, {});
    }
  };

  const handleInstallSubmit = () => {
    if (!installEntry) return;
    const missing = installEntry.required_env.filter(
      (item) => item.required && !(installEnv[item.name] ?? "").trim(),
    );
    if (missing.length > 0) {
      showToast(`${missing[0].prompt} required`, "error");
      return;
    }
    const envMap: Record<string, string> = {};
    Object.entries(installEnv).forEach(([k, v]) => {
      if (v.trim()) envMap[k] = v.trim();
    });
    void runInstall(installEntry, envMap);
  };

  // Put "Add Server" button in page header
  useLayoutEffect(() => {
    setEnd(
      <Button
        className="uppercase"
        size="sm"
        onClick={() => setCreateModalOpen(true)}
      >
        Add Server
      </Button>,
    );
    return () => {
      setEnd(null);
    };
  }, [setEnd, loading]);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-24">
        <Spinner className="text-2xl text-primary" />
      </div>
    );
  }

  const diagnosticsByName: Record<string, McpCatalogDiagnostic[]> = {};
  diagnostics.forEach((d) => {
    (diagnosticsByName[d.name] ??= []).push(d);
  });

  const anyOAuthCompleted = servers.some(
    (s) => s.auth === "oauth" && flows[s.name]?.status === "completed",
  );

  return (
    <div className="flex flex-col gap-6">
      <Toast toast={toast} />

      <DeleteConfirmDialog
        open={serverDelete.isOpen}
        onCancel={serverDelete.cancel}
        onConfirm={serverDelete.confirm}
        title="Remove MCP server"
        description={
          serverDelete.pendingId
            ? `"${truncateText(serverDelete.pendingId, 40)}" — this will remove the server.`
            : "This will remove the server."
        }
        loading={serverDelete.isDeleting}
      />

      {/* Add server modal */}
      {createModalOpen && (
        <div
          ref={createModalRef}
          className="fixed inset-0 z-[100] flex items-center justify-center bg-background/85 p-4"
          onClick={(e) =>
            e.target === e.currentTarget && setCreateModalOpen(false)
          }
          role="dialog"
          aria-modal="true"
          aria-labelledby="create-mcp-title"
        >
          <div
            className={cn(
              themedBody,
              "relative w-full max-w-lg border border-border bg-card shadow-2xl flex flex-col",
            )}
          >
            <Button
              ghost
              size="icon"
              onClick={() => setCreateModalOpen(false)}
              className="absolute right-2 top-2 text-muted-foreground hover:text-foreground"
              aria-label="Close"
            >
              <X />
            </Button>

            <header className="p-5 pb-3 border-b border-border">
              <h2
                id="create-mcp-title"
                className="font-mondwest text-display text-base tracking-wider"
              >
                Add MCP server
              </h2>
            </header>

            <div className="p-5 grid gap-4">
              <div className="grid gap-2">
                <Label htmlFor="mcp-name">Name</Label>
                <Input
                  id="mcp-name"
                  autoFocus
                  placeholder="my-server"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                />
              </div>

              <div className="grid gap-2">
                <Label htmlFor="mcp-transport">Transport</Label>
                <Select
                  id="mcp-transport"
                  value={transport}
                  onValueChange={(v) => setTransport(v as Transport)}
                >
                  <SelectOption value="http">HTTP/SSE</SelectOption>
                  <SelectOption value="stdio">stdio</SelectOption>
                </Select>
              </div>

              {transport === "http" ? (
                <div className="grid gap-2">
                  <Label htmlFor="mcp-url">URL</Label>
                  <Input
                    id="mcp-url"
                    placeholder="https://example.com/mcp"
                    value={url}
                    onChange={(e) => setUrl(e.target.value)}
                  />
                </div>
              ) : (
                <>
                  <div className="grid gap-2">
                    <Label htmlFor="mcp-command">Command</Label>
                    <Input
                      id="mcp-command"
                      placeholder="npx"
                      value={command}
                      onChange={(e) => setCommand(e.target.value)}
                    />
                  </div>
                  <div className="grid gap-2">
                    <Label htmlFor="mcp-args">Args</Label>
                    <Input
                      id="mcp-args"
                      placeholder="-y @modelcontextprotocol/server-foo"
                      value={args}
                      onChange={(e) => setArgs(e.target.value)}
                    />
                  </div>
                </>
              )}

              <div className="grid gap-2">
                <Label htmlFor="mcp-env">Environment (KEY=VALUE per line)</Label>
                <textarea
                  id="mcp-env"
                  className="flex min-h-[80px] w-full border border-border bg-background/40 px-3 py-2 text-sm font-courier shadow-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-foreground/30 focus-visible:border-foreground/25"
                  placeholder={"API_KEY=secret\nDEBUG=1"}
                  value={env}
                  onChange={(e) => setEnv(e.target.value)}
                />
              </div>

              <div className="flex justify-end">
                <Button
                  className="uppercase"
                  size="sm"
                  onClick={handleCreate}
                  disabled={creating}
                  prefix={creating ? <Spinner /> : undefined}
                >
                  {creating ? "Adding..." : "Add"}
                </Button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Catalog install modal (required env vars) */}
      {installEntry && (
        <div
          ref={installModalRef}
          className="fixed inset-0 z-[100] flex items-center justify-center bg-background/85 p-4"
          onClick={(e) =>
            e.target === e.currentTarget && setInstallEntry(null)
          }
          role="dialog"
          aria-modal="true"
          aria-labelledby="install-mcp-title"
        >
          <div
            className={cn(
              themedBody,
              "relative w-full max-w-lg border border-border bg-card shadow-2xl flex flex-col",
            )}
          >
            <Button
              ghost
              size="icon"
              onClick={() => setInstallEntry(null)}
              className="absolute right-2 top-2 text-muted-foreground hover:text-foreground"
              aria-label="Close"
            >
              <X />
            </Button>

            <header className="p-5 pb-3 border-b border-border">
              <h2
                id="install-mcp-title"
                className="font-mondwest text-display text-base tracking-wider"
              >
                Install {installEntry.name}
              </h2>
            </header>

            <div className="p-5 grid gap-4">
              <p className="text-xs text-muted-foreground">
                This MCP requires the following values to be configured.
              </p>
              {installEntry.required_env.map((item) => (
                <div className="grid gap-2" key={item.name}>
                  <Label htmlFor={`install-env-${item.name}`}>
                    {item.prompt}
                    {item.required ? " *" : ""}
                  </Label>
                  <Input
                    id={`install-env-${item.name}`}
                    type="password"
                    placeholder={item.name}
                    value={installEnv[item.name] ?? ""}
                    onChange={(e) =>
                      setInstallEnv((prev) => ({
                        ...prev,
                        [item.name]: e.target.value,
                      }))
                    }
                  />
                </div>
              ))}

              <div className="flex justify-end">
                <Button
                  className="uppercase"
                  size="sm"
                  onClick={handleInstallSubmit}
                  disabled={installingName === installEntry.name}
                  prefix={
                    installingName === installEntry.name ? (
                      <Spinner />
                    ) : undefined
                  }
                >
                  {installingName === installEntry.name
                    ? "Installing..."
                    : "Install"}
                </Button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Your MCP servers ── */}
      <div className="flex flex-col gap-3">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
          <H2
            variant="sm"
            className="flex items-center gap-2 text-muted-foreground"
          >
            <Server className="h-4 w-4" />
            Your MCP servers ({servers.length})
          </H2>
        </div>

        {restartNote && (
          <p className="text-xs text-warning">{restartNote}</p>
        )}

        {servers.length === 0 && (
          <Card>
            <CardContent className="py-8 text-center text-sm text-muted-foreground">
              No MCP servers configured.
            </CardContent>
          </Card>
        )}

        {servers.map((server) => {
          const envCount = Object.keys(server.env ?? {}).length;
          const result = testResults[server.name];
          const isOAuth = server.auth === "oauth";
          const flow = flows[server.name];
          const oauthStatus: McpOAuthStatus = flow?.status ?? "idle";
          const oauthInProgress =
            oauthStatus === "starting" || oauthStatus === "url_ready";
          const connectBusy = oauthInProgress || oauthBusy[server.name];

          return (
            <Card key={server.name}>
              <CardContent
                className={cn(
                  "flex flex-col gap-3 py-4",
                  !server.enabled && "opacity-60",
                )}
              >
                <div className="flex items-start gap-4">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1 flex-wrap">
                      <span className="font-medium text-sm truncate">
                        {server.name}
                      </span>
                      <Badge
                        tone={TRANSPORT_TONE[server.transport] ?? "secondary"}
                      >
                        {server.transport}
                      </Badge>
                      {!server.enabled && (
                        <Badge tone="outline">disabled</Badge>
                      )}
                      {isOAuth &&
                        (server.has_tokens ? (
                          <Badge
                            tone="outline"
                            className="text-emerald-500 border-emerald-500/40"
                          >
                            <ShieldCheck className="h-3 w-3 mr-1" />
                            tokens cached
                          </Badge>
                        ) : (
                          <Badge
                            tone="outline"
                            className="text-amber-500 border-amber-500/40"
                          >
                            <ShieldOff className="h-3 w-3 mr-1" />
                            not authenticated
                          </Badge>
                        ))}
                      {isOAuth && oauthStatusBadge(oauthStatus)}
                    </div>
                    <div className="flex items-center gap-4 text-xs text-muted-foreground">
                      {server.transport === "http" ? (
                        <span className="font-mono truncate">
                          {server.url ?? "—"}
                        </span>
                      ) : (
                        <span className="font-mono truncate">
                          {[server.command, ...(server.args ?? [])]
                            .filter(Boolean)
                            .join(" ") || "—"}
                        </span>
                      )}
                      {envCount > 0 && (
                        <span>
                          {envCount} env var{envCount === 1 ? "" : "s"}
                        </span>
                      )}
                    </div>
                    {result && (
                      <div className="mt-2 text-xs">
                        {result.ok ? (
                          <p className="text-success">
                            {result.tools.length === 0
                              ? "Connected — no tools"
                              : `Tools: ${result.tools
                                  .map((tool) => tool.name)
                                  .join(", ")}`}
                          </p>
                        ) : (
                          <p className="text-destructive">
                            {result.error ?? "Connection failed"}
                          </p>
                        )}
                      </div>
                    )}
                  </div>

                  <div className="flex items-center gap-1 shrink-0">
                    {isOAuth && (
                      <>
                        <Button
                          size="sm"
                          onClick={() => handleConnect(server.name)}
                          disabled={connectBusy}
                          prefix={connectBusy ? <Spinner /> : <Plug />}
                        >
                          {server.has_tokens ? "Reconnect" : "Connect"}
                        </Button>
                        {(oauthStatus === "completed" ||
                          oauthStatus === "failed") && (
                          <Button
                            ghost
                            size="sm"
                            onClick={() => handleResetOAuth(server.name)}
                          >
                            Reset
                          </Button>
                        )}
                      </>
                    )}
                    <Button
                      ghost
                      size="sm"
                      title={server.enabled ? "Disable" : "Enable"}
                      aria-label={server.enabled ? "Disable" : "Enable"}
                      onClick={() => handleToggleEnabled(server)}
                      disabled={togglingName === server.name}
                      prefix={
                        togglingName === server.name ? (
                          <Spinner />
                        ) : (
                          <Power />
                        )
                      }
                      className={server.enabled ? "text-success" : undefined}
                    >
                      {server.enabled ? "Disable" : "Enable"}
                    </Button>

                    <Button
                      ghost
                      size="icon"
                      title="Test connection"
                      aria-label="Test connection"
                      onClick={() => handleTest(server)}
                      disabled={testing === server.name}
                    >
                      {testing === server.name ? <Spinner /> : <Zap />}
                    </Button>

                    <Button
                      ghost
                      destructive
                      size="icon"
                      title="Delete"
                      aria-label="Delete"
                      onClick={() => serverDelete.requestDelete(server.name)}
                    >
                      <Trash2 />
                    </Button>
                  </div>
                </div>

                {/* OAuth flow detail for the browser-driven handshake. */}
                {isOAuth && oauthStatus === "url_ready" && flow?.url && (
                  <div className="rounded bg-blue-500/10 border border-blue-500/30 p-3 text-sm">
                    <p className="font-medium mb-1">Open this URL to authorize:</p>
                    <a
                      href={flow.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex items-center gap-1 text-blue-400 hover:text-blue-300 break-all"
                    >
                      {flow.url}
                      <ExternalLink className="h-3 w-3 shrink-0" />
                    </a>
                    <p className="text-xs text-muted-foreground mt-2">
                      The provider will redirect back to{" "}
                      <code className="text-xs">/oauth/callback</code> on this
                      dashboard. Status updates automatically.
                    </p>
                  </div>
                )}

                {isOAuth && oauthStatus === "starting" && (
                  <div className="text-xs text-muted-foreground flex items-center gap-2">
                    <Spinner />
                    Contacting{" "}
                    <code className="text-xs">{server.url ?? server.name}</code>…
                  </div>
                )}

                {isOAuth && oauthStatus === "completed" && (
                  <div className="rounded bg-emerald-500/10 border border-emerald-500/30 p-3 text-sm flex items-center gap-2">
                    <CheckCircle2 className="h-4 w-4 text-emerald-500 shrink-0" />
                    <span>
                      Connected — discovered {flow?.tool_count ?? 0} tools.
                      Restart the gateway so it picks up the cached tokens.
                    </span>
                  </div>
                )}

                {isOAuth && oauthStatus === "failed" && flow?.error && (
                  <div className="rounded bg-rose-500/10 border border-rose-500/30 p-3 text-sm flex items-start gap-2">
                    <XCircle className="h-4 w-4 text-rose-500 shrink-0 mt-0.5" />
                    <span className="break-words">{flow.error}</span>
                  </div>
                )}
              </CardContent>
            </Card>
          );
        })}

        {anyOAuthCompleted && (
          <div className="pt-1">
            <Button
              onClick={handleRestartGateway}
              disabled={restarting}
              prefix={restarting ? <Spinner /> : <RotateCw />}
            >
              Restart gateway
            </Button>
            <p className="text-xs text-muted-foreground mt-2">
              Once restarted the gateway reconnects to any server whose tokens
              are now cached, and its tools become available to the agent.
            </p>
          </div>
        )}
      </div>

      {/* ── Catalog ── */}
      <div className="flex flex-col gap-3">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
          <H2
            variant="sm"
            className="flex items-center gap-2 text-muted-foreground"
          >
            <Package className="h-4 w-4" />
            Catalog ({catalog.length})
          </H2>
        </div>

        <p className="text-xs text-muted-foreground">
          Browse Nous-approved MCP servers and install them with one click.
        </p>

        {catalog.length === 0 && (
          <Card>
            <CardContent className="py-8 text-center text-sm text-muted-foreground">
              No catalog entries available.
            </CardContent>
          </Card>
        )}

        {catalog.map((entry) => {
          const entryDiags = diagnosticsByName[entry.name] ?? [];
          const isInstalling = installingName === entry.name;

          return (
            <Card key={entry.name}>
              <CardContent className="flex items-start gap-4 py-4">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1 flex-wrap">
                    <span className="font-medium text-sm truncate">
                      {entry.name}
                    </span>
                    <Badge
                      tone={TRANSPORT_TONE[entry.transport] ?? "secondary"}
                    >
                      {entry.transport}
                    </Badge>
                    <Badge tone="outline">auth: {entry.auth_type}</Badge>
                    {isHttpUrl(entry.source) ? (
                      <a
                        href={entry.source}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-xs text-primary underline underline-offset-2 hover:opacity-80"
                      >
                        source ↗
                      </a>
                    ) : (
                      entry.source && (
                        <Badge tone="outline">{entry.source}</Badge>
                      )
                    )}
                    {entry.installed && (
                      <Badge tone="success">Installed</Badge>
                    )}
                    {entry.installed && !entry.enabled && (
                      <Badge tone="outline">disabled</Badge>
                    )}
                  </div>
                  {entry.description && (
                    <p className="text-xs text-muted-foreground">
                      {entry.description}
                    </p>
                  )}
                  {/* Connection detail: what the agent actually talks to. */}
                  {entry.transport === "http" && entry.url && (
                    <p className="mt-1 text-xs text-muted-foreground">
                      <span className="font-medium">Endpoint:</span>{" "}
                      <code className="font-mono">{entry.url}</code>
                    </p>
                  )}
                  {entry.transport === "stdio" && entry.command && (
                    <p className="mt-1 text-xs text-muted-foreground break-all">
                      <span className="font-medium">Runs:</span>{" "}
                      <code className="font-mono">
                        {[entry.command, ...entry.args].join(" ")}
                      </code>
                    </p>
                  )}
                  {/* Git bootstrap — surfaced so users see what gets cloned/run
                      before they install (matches the docs trust model). */}
                  {entry.install_url && (
                    <p className="mt-1 text-xs text-muted-foreground break-all">
                      <span className="font-medium">Installs from:</span>{" "}
                      {isHttpUrl(entry.install_url) ? (
                        <a
                          href={entry.install_url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-primary underline underline-offset-2 hover:opacity-80"
                        >
                          {entry.install_url}
                        </a>
                      ) : (
                        <code className="font-mono">{entry.install_url}</code>
                      )}
                      {entry.install_ref && (
                        <span> @ {entry.install_ref}</span>
                      )}
                    </p>
                  )}
                  {entry.bootstrap.length > 0 && (
                    <details className="mt-1 text-xs text-muted-foreground">
                      <summary className="cursor-pointer select-none">
                        Bootstrap commands ({entry.bootstrap.length})
                      </summary>
                      <ul className="mt-1 ml-3 list-disc space-y-0.5">
                        {entry.bootstrap.map((cmd, i) => (
                          <li key={`${entry.name}-bs-${i}`} className="break-all">
                            <code className="font-mono">{cmd}</code>
                          </li>
                        ))}
                      </ul>
                    </details>
                  )}
                  {entry.post_install && (
                    <details className="mt-1 text-xs text-muted-foreground">
                      <summary className="cursor-pointer select-none">
                        Setup notes
                      </summary>
                      <p className="mt-1 whitespace-pre-wrap">
                        {entry.post_install.trim()}
                      </p>
                    </details>
                  )}
                  {entryDiags.map((d, i) => (
                    <p
                      key={`${entry.name}-diag-${i}`}
                      className="text-xs text-warning mt-1"
                    >
                      {d.message}
                    </p>
                  ))}
                </div>

                <div className="flex items-center gap-1 shrink-0">
                  {entry.installed ? (
                    <Badge tone="success">Installed</Badge>
                  ) : (
                    <Button
                      className="uppercase"
                      size="sm"
                      onClick={() => handleInstallClick(entry)}
                      disabled={isInstalling}
                      prefix={isInstalling ? <Spinner /> : undefined}
                    >
                      {isInstalling ? "Installing..." : "Install"}
                    </Button>
                  )}
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>
    </div>
  );
}
