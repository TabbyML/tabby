import { createThread, type ThreadOptions } from "./target";

/**
 * Creates a thread from a web worker. This function can be used either from a JavaScript
 * environment that *created* a web worker, or from within a web worker that has been
 * created.
 *
 * @see https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API/Using_web_workers
 *
 * @example
 * import {createThreadFromWebWorker} from '@quilted/threads';
 *
 * // If inside a web worker:
 * const thread = createThreadFromWebWorker(self);
 *
 * // If in an environment that creates a worker:
 * const worker = new Worker('worker.js');
 * const thread = createThreadFromWebWorker(worker);
 *
 * await thread.sendMessage('Hello world!');
 */
export function createThreadFromWebWorker<
  Self = Record<string, never>,
  Target = Record<string, never>,
>(worker: Worker, options?: ThreadOptions<Self, Target>) {
  return createThread(
    {
      send(...args: [any, Transferable[]]) {
        worker.postMessage(...args);
      },
      listen(listener, { signal }) {
        worker.addEventListener(
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
