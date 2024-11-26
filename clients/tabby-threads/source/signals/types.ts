import type {ThreadAbortSignal} from '../abort-signal.ts';

/**
 * A representation of a Preact signal that can be serialized between
 * two threads.
 */
export interface ThreadSignal<T> {
  /**
   * The initial value of the signal.
   */
  initial: T;

  /**
   * Sets the value of the original signal.
   */
  set?(value: T): void;

  /**
   * A function to connect the signal between the two threads. This
   * function should be called by the sibling thread when the abort
   * state changes. It must also respond with a boolean indicating
   * whether the original signal was aborted at the time the listener
   * was attached, which is used to detect a change in state that happened
   * during the message passing process.
   */
  start(
    listener: (value: T) => void,
    options?: {
      /**
       * An `AbortSignal` that can be used to stop synchronizing the signal
       * between the two threads.
       */
      signal?: AbortSignal | ThreadAbortSignal;
    },
  ): void;
}
