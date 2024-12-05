import { type Signal } from "@quilted/signals";
import { NestedAbortController } from "@quilted/events";

import { retain, release } from "../memory";
import { acceptThreadAbortSignal } from "../abort-signal";

import { type ThreadSignal } from "./types";

/**
 * Converts a Preact signal into a version of that signal that can
 * be transferred to a target `Thread`. On the paired thread, this
 * "thread-safe" version of the signal can be turned into an actual,
 * live Preact signal using `acceptThreadSignal()`.
 */
export function createThreadSignal<T>(
  signal: Signal<T>,
  {
    /**
     * Whether the thread signal should have a method to write a value
     * back to the original signal. This allows you to create two-way
     * synchronization between the two threads, which can be useful, but
     * can also be hard to reason about.
     *
     * @default false
     */
    writable = false,

    /**
     * An optional `AbortSignal` that can cancel synchronizing the
     * signal to its paired thread.
     */
    signal: teardownAbortSignal,
  }: { writable?: boolean; signal?: AbortSignal } = {}
): ThreadSignal<T> {
  let initialVersion: number;

  return {
    get initial() {
      // @see https://github.com/preactjs/signals/blob/main/mangle.json#L56
      initialVersion = (signal as any).i;
      return signal.peek();
    },
    set:
      writable && !isReadonlySignal(signal)
        ? (value) => {
            signal.value = value;
          }
        : undefined,
    start(subscriber, { signal: threadAbortSignal } = {}) {
      retain(subscriber);

      const abortSignal =
        threadAbortSignal && acceptThreadAbortSignal(threadAbortSignal);

      const finalAbortSignal =
        abortSignal && teardownAbortSignal
          ? new NestedAbortController(abortSignal, teardownAbortSignal).signal
          : abortSignal ?? teardownAbortSignal;

      const teardown = signal.subscribe((value: any) => {
        if ((signal as any).i === initialVersion) {
          return;
        }

        subscriber(value);
      });

      finalAbortSignal?.addEventListener("abort", () => {
        teardown();
        release(subscriber);
      });
    },
  };
}

function isReadonlySignal<T>(signal: Signal<T>): boolean {
  return (
    Object.getOwnPropertyDescriptor(Object.getPrototypeOf(signal), "value")
      ?.set == null
  );
}
