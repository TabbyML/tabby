import { CustomAbortSignal } from "../../completions/context/utils";
import { AbortError, TimeoutError } from "../../completions/errors";

type PromiseCreator<T> = () => Promise<T>;
interface Queued<T> {
  creator: PromiseCreator<T>;
  abortSignal?: AbortSignal | CustomAbortSignal;
  resolve: (value: T) => void;
  reject: (reason: Error) => void;
}

export type Limiter = <T>(
  creator: PromiseCreator<T>,
  abortSignal?: AbortSignal | CustomAbortSignal
) => Promise<T>;

export function createLimiter(limit: number, timeout: number): Limiter {
  const queue: Queued<unknown>[] = [];
  let inflightPromises = 0;

  function processNext(): void {
    if (inflightPromises >= limit) {
      return;
    }

    if (queue.length === 0) {
      return;
    }

    const next = queue.shift()!;
    inflightPromises += 1;

    let didTimeout = false;

    const timeoutId = setTimeout(() => {
      didTimeout = true;
      next.reject(new TimeoutError());
      inflightPromises -= 1;
      processNext();
    }, timeout);

    const runner = next.creator();
    runner
      .then((value) => {
        if (didTimeout) {
          return;
        }
        next.resolve(value);
      })
      .catch((error) => {
        if (didTimeout) {
          return;
        }
        next.reject(error);
      })
      .finally(() => {
        if (didTimeout) {
          return;
        }
        clearTimeout(timeoutId);
        inflightPromises -= 1;
        processNext();
      });
  }

  return function enqueue<T>(
    creator: () => Promise<T>,
    abortSignal?: AbortSignal | CustomAbortSignal
  ): Promise<T> {
    let queued: Queued<T>;
    const promise = new Promise<T>((resolve, reject) => {
      queued = {
        creator,
        abortSignal,
        resolve,
        reject,
      };
    });
    queue.push(queued! as Queued<unknown>);
    abortSignal?.addEventListener("abort", () => {
      // Only abort queued requests
      const index = queue.indexOf(queued! as Queued<unknown>);
      if (index < 0) {
        return;
      }

      queued.reject(new AbortError());
      queue.splice(index, 1);
    });

    processNext();

    return promise;
  };
}
