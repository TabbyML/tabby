/**
 * Creates a simple subscriber that can be used to register callbacks
 */
type Listener<T> = (value: T) => void;
interface Subscriber<T> {
  subscribe(listener: Listener<T>): () => void;
  notify(value: T): void;
}
export function createSubscriber<T>(): Subscriber<T> {
  const listeners: Set<Listener<T>> = new Set();
  function subscribe(listener: Listener<T>): () => void {
    listeners.add(listener);
    return () => listeners.delete(listener);
  }

  function notify(value: T): void {
    for (const listener of listeners) {
      listener(value);
    }
  }

  return {
    subscribe,
    notify,
  };
}
