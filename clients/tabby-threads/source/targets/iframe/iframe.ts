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
export function createThreadFromIframe<
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

  const sendMessage: ThreadTarget["send"] = function send(message, transfer) {
    iframe.contentWindow?.postMessage(message, targetOrigin, transfer);
  };

  const connectedPromise = new Promise<void>((resolve) => {
    const abort = options.signal
      ? new NestedAbortController(options.signal)
      : new AbortController();

    window.addEventListener(
      "message",
      (event) => {
        if (event.source !== iframe.contentWindow) return;

        if (event.data === RESPONSE_MESSAGE) {
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
        resolve();
      },
      { once: true }
    );

    sendMessage(CHECK_MESSAGE);
  });

  return createThread(
    {
      send(message, transfer) {
        if (!connected) {
          return connectedPromise.then(() => {
            if (connected) return sendMessage(message, transfer);
          });
        }

        return sendMessage(message, transfer);
      },
      listen(listen, { signal }) {
        self.addEventListener(
          "message",
          (event) => {
            if (event.source !== iframe.contentWindow) return;
            if (event.data === RESPONSE_MESSAGE) return;
            listen(event.data);
          },
          { signal }
        );
      },
    },
    options
  );
}
