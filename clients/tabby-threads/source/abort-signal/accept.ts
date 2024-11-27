import { retain, release } from "../memory";
import type { ThreadAbortSignal } from "./types";

/**
 * Call this function in a thread receiving a `ThreadAbortSignal` to
 * turn it into a "live" `AbortSignal`. The resulting signal will
 * connect the thread to its sending pair, and will abort it when the
 * original `AbortSignal` is aborted.
 */
export function acceptThreadAbortSignal(
  signal: AbortSignal | ThreadAbortSignal
): AbortSignal {
  if (isAbortSignal(signal)) return signal;

  const abort = new AbortController();

  const { aborted, start } = signal;

  if (aborted) {
    abort.abort();
    return abort.signal;
  }

  if (start) {
    retain(start);
    start((aborted) => {
      if (aborted) abort.abort();
    });
  }

  abort.signal.addEventListener(
    "abort",
    async () => {
      release(start);
    },
    { once: true }
  );

  return abort.signal;
}

function isAbortSignal(value?: unknown): value is AbortSignal {
  return (
    value != null &&
    (value as any).aborted != null &&
    typeof (value as any).addEventListener === "function"
  );
}
