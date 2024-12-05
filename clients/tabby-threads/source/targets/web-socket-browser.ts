import { createThread, type ThreadOptions } from "./target";

/**
 * Creates a thread from a `WebSocket` instance in the browser.
 *
 * @see https://developer.mozilla.org/en-US/docs/Web/API/WebSocket
 *
 * @example
 * import {createThreadFromBrowserWebSocket} from '@quilted/threads';
 *
 * const websocket = new WebSocket('ws://localhost:8080');
 * const thread = createThreadFromBrowserWebSocket(websocket);
 * await thread.sendMessage('Hello world!');
 */
export function createThreadFromBrowserWebSocket<
  Self = Record<string, never>,
  Target = Record<string, never>,
>(websocket: WebSocket, options?: ThreadOptions<Self, Target>) {
  return createThread(
    {
      async send(message) {
        if (websocket.readyState !== websocket.OPEN) {
          await new Promise<void>((resolve) => {
            websocket.addEventListener(
              "open",
              () => {
                resolve();
              },
              { once: true }
            );
          });
        }

        websocket.send(JSON.stringify(message));
      },
      listen(listener, { signal }) {
        websocket.addEventListener(
          "message",
          (event) => {
            listener(JSON.parse(event.data));
          },
          { signal }
        );
      },
    },
    options
  );
}
