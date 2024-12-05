// Polyfill for AbortSignal.any(signals) which added in Node.js v20.
export function abortSignalFromAnyOf(signals: (AbortSignal | undefined)[]) {
  const controller = new AbortController();
  for (const signal of signals) {
    if (signal?.aborted) {
      controller.abort(signal.reason);
      return signal;
    }
    signal?.addEventListener("abort", () => controller.abort(signal.reason), {
      signal: controller.signal,
    });
  }
  return controller.signal;
}
