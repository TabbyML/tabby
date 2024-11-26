import { signal as createSignal, type Signal } from "@quilted/signals";

import { createThreadAbortSignal } from "../abort-signal";

import { type ThreadSignal } from "./types";

/**
 * Call this function in a thread receiving a `ThreadSignal` to
 * turn it into a "live" Preact signal. The resulting signal will
 * connect the thread to its sending pair, and will update it as the
 * signal value changes. If the thread signal is writable, writing
 * the value of the resulting signal will also update it on the paired
 * thread.
 */
export function acceptThreadSignal<T>(
  threadSignal: ThreadSignal<T>,
  {
    signal: abortSignal,
  }: {
    /**
     * An optional `AbortSignal` that can cancel synchronizing the
     * signal to its paired thread.
     */
    signal?: AbortSignal;
  } = {}
): Signal<T> {
  const signal = createSignal(threadSignal.initial);
  const threadAbortSignal = abortSignal && createThreadAbortSignal(abortSignal);

  const valueDescriptor = Object.getOwnPropertyDescriptor(
    Object.getPrototypeOf(signal),
    "value"
  )!;

  Object.defineProperty(signal, "value", {
    ...valueDescriptor,
    get() {
      return valueDescriptor.get?.call(this);
    },
    set(value) {
      if (threadSignal.set == null) {
        throw new Error(`You can’t set the value of a readonly thread signal.`);
      }

      threadSignal.set(value);
      return valueDescriptor.set?.call(this, value);
    },
  });

  threadSignal.start(
    (value) => {
      valueDescriptor.set?.call(signal, value);
    },
    { signal: threadAbortSignal }
  );

  return signal;
}

/**
 * Returns `true` if the passed object is a `ThreadSignal`.
 */
export function isThreadSignal<T = unknown>(
  value?: unknown
): value is ThreadSignal<T> {
  return (
    value != null &&
    typeof value === "object" &&
    "initial" in value &&
    typeof (value as any).start === "function"
  );
}
