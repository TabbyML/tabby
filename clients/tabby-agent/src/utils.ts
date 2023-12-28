export function splitLines(input: string) {
  const lines = input.match(/.*(?:$|\r?\n)/g)?.filter(Boolean) ?? []; // Split lines and keep newline character
  if (lines.length > 0 && lines[lines.length - 1]?.endsWith("\n")) {
    // Keep last empty line
    lines.push("");
  }
  return lines;
}

export function splitWords(input: string) {
  return input.match(/\w+|\W+/g)?.filter(Boolean) ?? []; // Split consecutive words and non-words
}

export function isBlank(input: string) {
  return input.trim().length === 0;
}

// Indentation

export function getIndentationLevel(line: string, indentation?: string) {
  if (indentation === undefined) {
    return line.match(/^[ \t]*/)?.[0]?.length ?? 0;
  } else if (indentation === "\t") {
    return line.match(/^\t*/g)?.[0].length ?? 0;
  } else if (indentation.match(/^ *$/)) {
    const spaces = line.match(/^ */)?.[0].length ?? 0;
    return spaces / indentation.length;
  } else {
    throw new Error(`Invalid indentation: ${indentation}`);
  }
}

// function foo(a) {  // <-- block opening line
//   return a;
// }                  // <-- block closing line
export function isBlockOpeningLine(lines: string[], index: number): boolean {
  if (index < 0 || index >= lines.length - 1) {
    return false;
  }
  return getIndentationLevel(lines[index]!) < getIndentationLevel(lines[index + 1]!);
}

export function isBlockClosingLine(lines: string[], index: number): boolean {
  if (index <= 0 || index > lines.length - 1) {
    return false;
  }
  return getIndentationLevel(lines[index - 1]!) > getIndentationLevel(lines[index]!);
}

// Auto-closing chars

export const autoClosingPairs = [
  ["(", ")"],
  ["[", "]"],
  ["{", "}"],
  ["'", "'"],
  ['"', '"'],
  ["`", "`"],
];

export const autoClosingPairOpenings = autoClosingPairs.map((pair) => pair[0]);
export const autoClosingPairClosings = autoClosingPairs.map((pair) => pair[1]);

// FIXME: This function is not good enough, it can not handle escaped characters.
export function findUnpairedAutoClosingChars(input: string): string {
  const stack: string[] = [];

  for (const char of input) {
    [
      ["(", ")"],
      ["[", "]"],
      ["{", "}"],
    ].forEach((pair) => {
      if (char === pair[1]) {
        if (stack.length > 0 && stack[stack.length - 1] === pair[0]) {
          stack.pop();
        } else {
          stack.push(char);
        }
      }
    });
    if ("([{".includes(char)) {
      stack.push(char);
    }
    ["'", '"', "`"].forEach((quote) => {
      if (char === quote) {
        if (stack.length > 0 && stack.includes(quote)) {
          stack.splice(stack.lastIndexOf(quote), stack.length - stack.lastIndexOf(quote));
        } else {
          stack.push(char);
        }
      }
    });
  }
  return stack.join("");
}

// Using string levenshtein distance is not good, because variable name may create a large distance.
// Such as distance is 9 between `const fooFooFoo = 1;` and `const barBarBar = 1;`, but maybe 1 is enough.
// May be better to count distance based on words instead of characters.
import * as levenshtein from "fast-levenshtein";
export function calcDistance(a: string, b: string) {
  return levenshtein.get(a, b);
}

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

// Http Error
export class HttpError extends Error {
  public readonly status: number;
  public readonly statusText: string;
  public readonly response: Response;

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
    (error instanceof HttpError && [408, 499].includes(error.status))
  );
}

export function isCanceledError(error: any) {
  return error instanceof Error && error.name === "AbortError";
}

export function errorToString(error: Error & { cause?: Error }) {
  let message = error.message || error.toString();
  if (error.cause) {
    message += "\nCaused by: " + errorToString(error.cause);
  }
  return message;
}
