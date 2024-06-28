import { createThread, type ThreadOptions } from "@quilted/threads";
import type { ServerApi, ClientApi } from "tabby-chat-panel";
import { WebviewView } from "vscode";

export function createThreadFromWebview<Self = Record<string, never>, Target = Record<string, never>>(
  webview: WebviewView,
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

export function createClient(webview: WebviewView, api: ClientApi): ServerApi {
  return createThreadFromWebview(webview, {
    expose: {
      navigate: api.navigate,
      refresh: api.refresh,
    },
  });
}
