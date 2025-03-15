import type { Webview } from "vscode";
import {
  type ServerApiList,
  type ClientApi,
  type ServerApi,
  createClient as createClientFromThread,
} from "tabby-chat-panel";
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

export async function createClient(webview: Webview, api: ClientApi): Promise<ServerApiList> {
  const thread = createThreadFromWebview<ClientApi, ServerApi>(webview, { expose: api });
  return await createClientFromThread(thread);
}
