import { createThread, type ThreadOptions } from "./target";

/**
 * Creates a thread from a `BroadcastChannel` instance in the browser.
 *
 * @see https://developer.mozilla.org/en-US/docs/Web/API/BroadcastChannel
 *
 * @example
 * import {createThreadFromBroadcastChannel} from '@quilted/threads';
 *
 * const channel = new BroadcastChannel('my-channel');;
 * const thread = createThreadFromBroadcastChannel(channel);
 * await thread.sendMessage('Hello world!');
 */
export function createThreadFromBroadcastChannel<
  Self = Record<string, never>,
  Target = Record<string, never>,
>(channel: BroadcastChannel, options?: ThreadOptions<Self, Target>) {
  return createThread(
    {
      send(message) {
        channel.postMessage(message);
      },
      listen(listener, { signal }) {
        channel.addEventListener(
          "message",
          (event) => {
            listener(event.data);
          },
          { signal }
        );
      },
    },
    options
  );
}
