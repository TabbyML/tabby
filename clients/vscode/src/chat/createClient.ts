import type { Webview } from "vscode";
import type { ServerApi, ClientApiMethods } from "tabby-chat-panel";
import { createThread, type ThreadOptions } from "tabby-threads";
import { getLogger } from "../logger";

async function createThreadFromWebview<Self = Record<string, never>, Target = Record<string, never>>(
  webview: Webview,
  options?: ThreadOptions<Self, Target>,
) {
  getLogger().info("Creating thread from webview");
  const thread = await createThread(
    {
      send(message) {
        getLogger().debug("Sending message to chat panel", message);
        webview.postMessage({ action: "postMessageToChatPanel", message });
      },
      listen(listener, { signal }) {
        getLogger().debug("Setting up message listener");
        const { dispose } = webview.onDidReceiveMessage((msg) => {
          getLogger().debug("Received message from chat panel", msg);
          listener(msg);
        });
        signal?.addEventListener("abort", () => {
          getLogger().debug("Disposing message listener");
          dispose();
        });
      },
    },
    options,
  );
  getLogger().info("Thread created");
  return thread;
}

export async function createClient(webview: Webview, api: ClientApiMethods): Promise<ServerApi> {
  const logger = getLogger();
  logger.info("Creating client with exposed methods:", Object.keys(api));
  const thread = await createThreadFromWebview(webview, {
    expose: {
      refresh: api.refresh,
      onApplyInEditor: api.onApplyInEditor,
      onApplyInEditorV2: api.onApplyInEditorV2,
      onLoaded: api.onLoaded,
      onCopy: api.onCopy,
      onKeyboardEvent: api.onKeyboardEvent,
      lookupSymbol: api.lookupSymbol,
      openInEditor: api.openInEditor,
      openExternal: api.openExternal,
      readWorkspaceGitRepositories: api.readWorkspaceGitRepositories,
      getActiveEditorSelection: api.getActiveEditorSelection,
      fetchSessionState: api.fetchSessionState,
      storeSessionState: api.storeSessionState,
      listFileInWorkspace: api.listFileInWorkspace,
      readFileContent: api.readFileContent,
    },
  });
  getLogger().info("Client created");
  return thread as unknown as ServerApi;
}
