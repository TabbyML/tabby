/**
 * A representation of an `AbortSignal` that can be serialized between
 * two threads.
 */
export interface ThreadAbortSignal {
  /**
   * Whether the signal was already aborted at the time it was
   * sent to the sibling thread.
   */
  aborted: boolean;

  /**
   * A function to connect the signal between the two threads. This
   * function should be called by the sibling thread when the abort
   * state changes (including changes since the thread-safe abort signal
   * was created).
   */
  start?(listener: (aborted: boolean) => void): void;
}
