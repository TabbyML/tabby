import { createThread, type ThreadOptions } from "./target";

/**
 * Creates a thread from a `MessagePort` instance in the browser.
 *
 * @see https://developer.mozilla.org/en-US/docs/Web/API/MessagePort
 *
 * @example
 * import {createThreadFromMessagePort} from '@quilted/threads';
 *
 * const channel = new MessageChannel();
 * const threadOne = createThreadFromMessagePort(channel.port1);
 * const threadTwo = createThreadFromMessagePort(channel.port2, {
 *   expose: {
 *     sendMessage: (message) => console.log(message),
 *   },
 * });
 *
 * await threadOne.sendMessage('Hello world!');
 */
export function createThreadFromMessagePort<
  Self = Record<string, never>,
  Target = Record<string, never>,
>(port: MessagePort, options?: ThreadOptions<Self, Target>) {
  return createThread(
    {
      send(...args: [any, Transferable[]]) {
        port.postMessage(...args);
      },
      listen(listener, { signal }) {
        port.addEventListener(
          "message",
          (event) => {
            listener(event.data);
          },
          { signal }
        );

        port.start();
      },
    },
    options
  );
}
