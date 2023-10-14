export function splitLines(input: string) {
  const lines = input.match(/.*(?:$|\r?\n)/g).filter(Boolean); // Split lines and keep newline character
  if (lines.length > 0 && lines[lines.length - 1].endsWith("\n")) { // Keep last empty line
    lines.push("");
  }
  return lines;
}

export function splitWords(input: string) {
  return input.match(/\w+|\W+/g).filter(Boolean); // Split consecutive words and non-words
}

export function isBlank(input: string) {
  return input.trim().length === 0;
}

// Using string levenshtein distance is not good, because variable name may create a large distance.
// Such as distance is 9 between `const fooFooFoo = 1;` and `const barBarBar = 1;`, but maybe 1 is enough.
// May be better to count distance based on words instead of characters.
import * as levenshtein from "fast-levenshtein";
export function calcDistance(a: string, b: string) {
  return levenshtein.get(a, b);
}

// Polyfill for AbortSignal.any(signals) which added in Node.js v20.
export function abortSignalFromAnyOf(signals: AbortSignal[]) {
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

// Http Error
export class HttpError extends Error {
  status: number;
  statusText: string;
  response: Response;

  constructor(response: Response) {
    super(`${response.status} ${response.statusText}`);
    this.name = "HttpError";
    this.status = response.status;
    this.statusText = response.statusText;
    this.response = response;
  }
}

export function isTimeoutError(error: any) {
  return (
    (error instanceof Error && error.name === "TimeoutError") ||
    (error instanceof HttpError && [408, 499].indexOf(error.status) !== -1)
  );
}

export function isCanceledError(error: any) {
  return error instanceof Error && error.name === "AbortError";
}
