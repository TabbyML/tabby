import type { Webview } from "vscode";
import {
  type ServerApiList,
  type ClientApi,
  type ServerApi,
  createClient as createClientFromThread,
} from "tabby-chat-panel";
import { createThread, type ThreadOptions, type ThreadTarget } from "tabby-threads";

// See "tabby-threads/source/targets/iframe/shared.ts"
export const CHECK_MESSAGE = "quilt.threads.ping";
export const RESPONSE_MESSAGE = "quilt.threads.pong";

class NestedAbortController extends AbortController {
  constructor(...parents: AbortSignal[]) {
    super();

    const abortedSignal = parents.find((signal) => signal.aborted);

    if (abortedSignal) {
      this.abort(abortedSignal.reason);
    } else {
      const abort = (event: Event) => this.abort((event.target as AbortSignal).reason);
      const options = { signal: this.signal };

      for (const signal of parents) {
        signal.addEventListener("abort", abort, options);
      }
    }
  }
}

// See "tabby-threads/source/targets/iframe/iframe.ts"
function createThreadFromWebview<Self = Record<string, never>, Target = Record<string, never>>(
  webview: Webview,
  options?: ThreadOptions<Self, Target>,
) {
  let connected = false;

  const sendMessage: ThreadTarget["send"] = function send(message: unknown) {
    webview.postMessage({ action: "postMessageToChatPanel", message });
  };

  const connectedPromise = new Promise<void>((resolve) => {
    const abort = options?.signal ? new NestedAbortController(options.signal) : new AbortController();

    const { dispose } = webview.onDidReceiveMessage((event) => {
      if (event === RESPONSE_MESSAGE) {
        connected = true;
        abort.abort();
        resolve();
      }
    });

    abort.signal.addEventListener(
      "abort",
      () => {
        dispose();
        resolve();
      },
      { once: true },
    );

    sendMessage(CHECK_MESSAGE);
  });

  return createThread(
    {
      send(message) {
        if (!connected) {
          return connectedPromise.then(() => {
            if (connected) {
              return sendMessage(message);
            }
          });
        }

        return sendMessage(message);
      },
      listen(listen, { signal }) {
        const { dispose } = webview.onDidReceiveMessage((event) => {
          if (event === RESPONSE_MESSAGE) {
            return;
          }
          listen(event);
        });
        signal?.addEventListener(
          "abort",
          () => {
            dispose();
          },
          { once: true },
        );
      },
    },
    options,
  );
}

export async function createClient(webview: Webview, api: ClientApi): Promise<ServerApiList> {
  const thread = createThreadFromWebview<ClientApi, ServerApi>(webview, { expose: api });
  return await createClientFromThread(thread);
}
