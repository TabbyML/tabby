import { createThread, type ThreadOptions } from "@quilted/threads";
import type { ServerApi, ClientApi } from "tabby-chat-panel";
import { WebviewView, WebviewPanel } from "vscode";

export function createThreadFromWebview<Self = Record<string, never>, Target = Record<string, never>>(
  webview: WebviewView | WebviewPanel,
  options?: ThreadOptions<Self, Target>,
) {
  return createThread(
    {
      send(...args) {
        webview.webview.postMessage({ data: args });
      },
      listen(listen, { signal }) {
        const { dispose } = webview.webview.onDidReceiveMessage((data) => {
          listen(data);
        });

        signal?.addEventListener("abort", () => {
          dispose();
        });
      },
    },
    options,
  );
}

export function createClient(webview: WebviewView | WebviewPanel, api: ClientApi): ServerApi {
  return createThreadFromWebview(webview, {
    expose: {
      navigate: api.navigate,
      refresh: api.refresh,
      onSubmitMessage: api.onSubmitMessage,
      onApplyInEditor: api.onApplyInEditor,
      onCopy: api.onCopy,
      onLoaded: api.onLoaded,
      focusOnEditor: api.focusOnEditor,
    },
  });
}
