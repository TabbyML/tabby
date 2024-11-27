import { retain, release } from "../memory";
import type { ThreadAbortSignal } from "./types.ts";

/**
 * Converts an `AbortSignal` into a version of that signal that can
 * be transferred to a target `Thread`. The resulting object can be
 * transferred to the paired thread, and turned into an actual `AbortSignal`
 * object using `acceptThreadAbortSignal()`.
 */
export function createThreadAbortSignal(
  signal: AbortSignal
): ThreadAbortSignal {
  const listeners = new Set<(aborted: boolean) => void>();

  if (signal.aborted) {
    return {
      aborted: true,
    };
  }

  signal.addEventListener(
    "abort",
    () => {
      for (const listener of listeners) {
        listener(signal.aborted);
        release(listener);
      }

      listeners.clear();
    },
    { once: true }
  );

  return {
    aborted: false,
    start(listener) {
      if (signal.aborted) {
        listener(true);
      } else {
        retain(listener);
        listeners.add(listener);
      }
    },
  };
}
