// Keywords appear in the code everywhere, but we don't want to use them for
// matching in code searching.
// Just filter them out before we start using a syntax parser.
const reservedKeywords = [
  // Typescript: https://github.com/microsoft/TypeScript/issues/2536
  "as",
  "any",
  "boolean",
  "break",
  "case",
  "catch",
  "class",
  "const",
  "constructor",
  "continue",
  "debugger",
  "declare",
  "default",
  "delete",
  "do",
  "else",
  "enum",
  "export",
  "extends",
  "false",
  "finally",
  "for",
  "from",
  "function",
  "get",
  "if",
  "implements",
  "import",
  "in",
  "instanceof",
  "interface",
  "let",
  "module",
  "new",
  "null",
  "number",
  "of",
  "package",
  "private",
  "protected",
  "public",
  "require",
  "return",
  "set",
  "static",
  "string",
  "super",
  "switch",
  "symbol",
  "this",
  "throw",
  "true",
  "try",
  "typeof",
  "var",
  "void",
  "while",
  "with",
  "yield",
];
export function extractNonReservedWordList(text: string): string {
  const re = /\w+/g;
  return [
    ...new Set(text.match(re)?.filter((symbol) => symbol.length > 2 && !reservedKeywords.includes(symbol))).values(),
  ].join(" ");
}

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
    return line.match(/^\t*/)?.[0].length ?? 0;
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
type AutoClosingCharPosition = "open" | "close" | "openOrClose";
type AutoClosingCharPattern = { chars: string; reg: RegExp };
type AutoClosingPairDifferent = { open: AutoClosingCharPattern; close: AutoClosingCharPattern };
type AutoClosingPairSame = { openOrClose: AutoClosingCharPattern };
type AutoClosingPair = AutoClosingPairDifferent | AutoClosingPairSame;

// FIXME: use syntax parser instead
export const autoClosingPairs: AutoClosingPair[] = [
  {
    open: {
      chars: "(",
      reg: /\(/,
    },
    close: {
      chars: ")",
      reg: /\)/,
    },
  },
  {
    open: {
      chars: "[",
      reg: /\[/,
    },
    close: {
      chars: "]",
      reg: /\]/,
    },
  },
  {
    open: {
      chars: "{",
      reg: /\{/,
    },
    close: {
      chars: "}",
      reg: /\}/,
    },
  },
  {
    open: {
      chars: "<",
      reg: /<(?=\w)/,
    },
    close: {
      chars: "/>",
      reg: /\/>/,
    },
  },
  {
    open: {
      chars: "<",
      reg: /<(?=[/\w])/,
    },
    close: {
      chars: ">",
      reg: />/,
    },
  },
  {
    openOrClose: {
      chars: '"',
      reg: /"/,
    },
  },
  {
    openOrClose: {
      chars: "'",
      reg: /'/,
    },
  },
  {
    openOrClose: {
      chars: "`",
      reg: /`/,
    },
  },
];

export const regOnlyAutoClosingCloseChars = /^([)\]}>"'`]|(\/>))*$/g;

// FIXME: This function is not good enough, it can not handle escaped characters.
export function findUnpairedAutoClosingChars(input: string): string[] {
  const stack: string[] = [];
  let index = 0;
  while (index < input.length) {
    const remain = input.slice(index);
    let nextFound: {
      index: number;
      found: { pair: AutoClosingPair; pos: AutoClosingCharPosition; pattern: AutoClosingCharPattern } | undefined;
    } = {
      index: remain.length,
      found: undefined,
    };
    autoClosingPairs.forEach((pair) => {
      Object.entries(pair).forEach(([pos, pattern]) => {
        const match = remain.match(pattern.reg);
        if (match && match.index !== undefined && match.index < nextFound.index) {
          nextFound = {
            index: match.index,
            found: { pair, pos: pos as AutoClosingCharPosition, pattern },
          };
        }
      });
    });
    if (!nextFound.found) {
      break;
    }
    switch (nextFound.found.pos) {
      case "openOrClose": {
        const chars = nextFound.found.pattern.chars;
        if (stack.length > 0 && stack.includes(chars)) {
          stack.splice(stack.lastIndexOf(chars), stack.length - stack.lastIndexOf(chars));
        } else {
          stack.push(chars);
        }
        break;
      }
      case "open": {
        stack.push(nextFound.found.pattern.chars);
        break;
      }
      case "close": {
        const pair = nextFound.found.pair;
        if (stack.length > 0 && "open" in pair && stack[stack.length - 1] === pair.open.chars) {
          stack.pop();
        } else {
          stack.push(nextFound.found.pattern.chars);
        }
        break;
      }
    }
    index += nextFound.index + nextFound.found.pattern.chars.length;
  }
  return stack;
}

// Using string levenshtein distance is not good, because variable name may create a large distance.
// Such as distance is 9 between `const fooFooFoo = 1;` and `const barBarBar = 1;`, but maybe 1 is enough.
// May be better to count distance based on words instead of characters.
import * as levenshtein from "fast-levenshtein";
export function calcDistance(a: string, b: string) {
  return levenshtein.get(a, b);
}

export function stringToRegExp(str: string): RegExp {
  const parts = /\/(.*)\/(.*)/.exec(str);
  if (parts && parts[1] && parts[2]) {
    return new RegExp(parts[1], parts[2]);
  }
  return new RegExp(str);
}

import type { Readable } from "readable-stream";
export async function parseChatResponse(readableStream: Readable): Promise<string> {
  let output = "";
  try {
    for await (const item of readableStream) {
      const delta = typeof item === "string" ? item : "";
      output += delta;
    }
  } catch (error) {
    if (error instanceof TypeError && error.message.startsWith("terminated")) {
      // ignore server side close error
    } else {
      throw error;
    }
  }
  return output;
}
