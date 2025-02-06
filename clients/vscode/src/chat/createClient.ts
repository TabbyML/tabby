import type { Webview } from "vscode";
import type { ServerApi, ClientApiMethods } from "tabby-chat-panel";
import { createThread, type ThreadOptions } from "tabby-threads";

function createThreadFromWebview<Self = Record<string, never>, Target = Record<string, never>>(
  webview: Webview,
  options?: ThreadOptions<Self, Target>,
) {
  return createThread(
    {
      send(message) {
        webview.postMessage({ action: "postMessageToChatPanel", message });
      },
      listen(listener, { signal }) {
        const { dispose } = webview.onDidReceiveMessage(listener);
        signal?.addEventListener("abort", () => {
          dispose();
        });
      },
    },
    options,
  );
}

export function createClient(webview: Webview, api: ClientApiMethods): ServerApi {
  return createThreadFromWebview(webview, {
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
      listSymbols: api.listSymbols,
    },
  });
}
