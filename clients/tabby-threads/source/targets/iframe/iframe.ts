import { NestedAbortController } from "@quilted/events";
import { createThread, type ThreadTarget, type ThreadOptions } from "../target";
import { CHECK_MESSAGE, RESPONSE_MESSAGE } from "./shared";

/**
 * Creates a thread from an iframe nested on a top-level document. To create
 * a thread from the contents of this iframe, use `createThreadFromInsideIframe()`
 * instead.
 *
 * @see https://developer.mozilla.org/en-US/docs/Web/HTML/Element/iframe
 *
 * @example
 * import {createThreadFromIframe} from '@quilted/threads';
 *
 * const iframe = document.createElement('iframe');
 * const thread = createThreadFromInsideIframe(iframe);
 * await thread.sendMessage('Hello world!');
 */
export async function createThreadFromIframe<
  Self = Record<string, never>,
  Target = Record<string, never>,
>(
  iframe: HTMLIFrameElement,
  {
    targetOrigin = "*",
    ...options
  }: ThreadOptions<Self, Target> & {
    /**
     * The target origin to use when sending `postMessage` events to the child frame.
     *
     * @see https://developer.mozilla.org/en-US/docs/Web/API/Window/postMessage#targetorigin
     * @default '*'
     */
    targetOrigin?: string;
  } = {}
) {
  let connected = false;
  console.log(
    "[createThreadFromIframe] Starting connection process with iframe:",
    iframe
  );

  const sendMessage: ThreadTarget["send"] = function send(message, transfer) {
    console.log("[createThreadFromIframe] Sending message:", message);
    iframe.contentWindow?.postMessage(message, targetOrigin, transfer);
  };

  const connectedPromise = new Promise<void>((resolve) => {
    const abort = options.signal
      ? new NestedAbortController(options.signal)
      : new AbortController();

    console.log("[createThreadFromIframe] Setting up message listener");
    window.addEventListener(
      "message",
      (event) => {
        if (event.source !== iframe.contentWindow) {
          console.log(
            "[createThreadFromIframe] Ignoring message from unknown source"
          );
          return;
        }

        console.log("[createThreadFromIframe] Received message:", event.data);
        if (event.data === RESPONSE_MESSAGE) {
          console.log(
            "[createThreadFromIframe] Received RESPONSE_MESSAGE, connection established"
          );
          connected = true;
          abort.abort();
          resolve();
        }
      },
      { signal: abort.signal }
    );

    abort.signal.addEventListener(
      "abort",
      () => {
        console.log("[createThreadFromIframe] Abort signal received");
        resolve();
      },
      { once: true }
    );

    console.log("[createThreadFromIframe] Sending CHECK_MESSAGE");
    sendMessage(CHECK_MESSAGE);
  });

  console.log("[createThreadFromIframe] Waiting for connection...");
  await connectedPromise;
  console.log(
    "[createThreadFromIframe] Connection established, creating thread"
  );

  const thread = await createThread(
    {
      send(message, transfer) {
        if (!connected) {
          console.log(
            "[createThreadFromIframe] Message queued until connection:",
            message
          );
          return connectedPromise.then(() => {
            if (connected) {
              console.log(
                "[createThreadFromIframe] Sending queued message:",
                message
              );
              return sendMessage(message, transfer);
            }
            console.log(
              "[createThreadFromIframe] Connection lost, message dropped:",
              message
            );
          });
        }

        return sendMessage(message, transfer);
      },
      listen(listen, { signal }) {
        console.log("[createThreadFromIframe] Setting up message listener");
        self.addEventListener(
          "message",
          (event) => {
            if (event.source !== iframe.contentWindow) {
              console.log(
                "[createThreadFromIframe] Ignoring message from unknown source"
              );
              return;
            }
            if (event.data === RESPONSE_MESSAGE) {
              console.log("[createThreadFromIframe] Ignoring RESPONSE_MESSAGE");
              return;
            }
            console.log(
              "[createThreadFromIframe] Received message:",
              event.data
            );
            listen(event.data);
          },
          { signal }
        );
      },
    },
    options
  );

  console.log("[createThreadFromIframe] Thread created successfully");
  return thread;
}
