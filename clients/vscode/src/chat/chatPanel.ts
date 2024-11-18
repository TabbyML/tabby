import { createThread, type ThreadOptions } from "@quilted/threads";
import type { ServerApi, ClientApi, ClientApiMethods } from "tabby-chat-panel";
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
  const hasCapability = (capability: ClientApiMethods): boolean => {
    return capability in exposedApi && typeof exposedApi[capability as keyof typeof exposedApi] === "function";
  };
  const exposedApi = {
    ...api,
    hasCapability: hasCapability,
  };
  return createThreadFromWebview(webview, {
    expose: exposedApi,
  });
}
