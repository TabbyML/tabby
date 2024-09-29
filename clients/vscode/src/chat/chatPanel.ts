import { createThread, type ThreadOptions } from "@quilted/threads";
import type { ServerApi, ClientApi } from "tabby-chat-panel";
import { Webview } from "vscode";

export function createThreadFromWebview<Self = Record<string, never>, Target = Record<string, never>>(
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

export function createClient(webview: Webview, api: ClientApi): ServerApi {
  return createThreadFromWebview(webview, {
    expose: {
      navigate: api.navigate,
      refresh: api.refresh,
      onSubmitMessage: api.onSubmitMessage,
      onApplyInEditor: api.onApplyInEditor,
      onCopy: api.onCopy,
      onLoaded: api.onLoaded,
      onKeyboardEvent: api.onKeyboardEvent,
    },
  });
}
