import { PostprocessFilter } from "./base";
import { CompletionItem } from "../solution";

function isOnlySpaces(str: string | null | undefined): boolean {
  if (!str) return true;
  return /^\s*$/.test(str);
}

function getLeadingSpaces(str: string): number {
  const match = str.match(/^(\s*)/);
  return match ? match[0].length : 0;
}

/**
 * normalizeIndentation postprocess filter.
 *
 * This function adjusts the indentation of code snippets (lines) based on the
 * current cursor's indentation context (context.currentLinePrefix). The primary
 * goal is to ensure that the inserted snippet aligns correctly with the surrounding code.
 *
 * How it works:
 * 1. If the current line prefix contains a tab, we assume the user is using tab-based indentation.
 *    - In this case, we only process the first line of the snippet: if its first character is a whitespace
 *      (either a space or a tab), we remove it. This simple adjustment helps avoid an extra indent.
 *
 * 2. If the current line prefix is space-indented (or contains only spaces) and is empty,
 *    we normalize the snippet further:
 *    - For the first snippet line: we calculate the total number of leading spaces by adding the
 *      cursor's prefix spaces to the snippet's first line spaces. If this sum is odd and the first
 *      line itself has an odd number of leading spaces, we remove all leading spaces from that line.
 *      This addresses cases where an "odd" indentation (i.e. not a multiple of the expected indent unit)
 *      would otherwise cause misalignment.
 *
 * 3. For every snippet line: if the line's leading spaces count is odd (and greater than zero),
 *    we remove one space to make the indentation even.
 *
 * The adjustments ensure that the snippet's indentation blends seamlessly with the user's current context,
 * preventing unintended extra indentation.
 */
export function normalizeIndentation(): PostprocessFilter {
  return (item: CompletionItem): CompletionItem => {
    const { context, lines } = item;
    if (!Array.isArray(lines) || lines.length === 0) return item;

    if (context.currentLinePrefix.includes("\t")) {
      const normalizedLines = [...lines];
      if (normalizedLines[0] && normalizedLines[0].length > 0) {
        if (/^[ \t]/.test(normalizedLines[0])) {
          normalizedLines[0] = normalizedLines[0].substring(1);
        }
      }
      return item.withText(normalizedLines.join(""));
    }

    if (!isOnlySpaces(context.currentLinePrefix)) {
      return item;
    }

    const normalizedLines = [...lines];
    const firstLine = normalizedLines[0];
    const cursorLineSpaces = getLeadingSpaces(context.currentLinePrefix);
    if (firstLine) {
      const firstLineSpaces = getLeadingSpaces(firstLine);
      // deal with current cursor odd indentation
      if ((firstLineSpaces + cursorLineSpaces) % 2 !== 0 && firstLineSpaces % 2 !== 0) {
        normalizedLines[0] = firstLine.substring(firstLineSpaces);
      }
    }

    //deal with extra space in the line indent
    for (let i = 0; i < normalizedLines.length; i++) {
      const line = normalizedLines[i];
      if (!line || !line.trim()) continue;
      const lineSpaces = getLeadingSpaces(line);
      if (lineSpaces > 0 && lineSpaces % 2 !== 0) {
        // move current line to recently close indent
        normalizedLines[i] = " ".repeat(lineSpaces - 1) + line.trimStart();
      }
    }

    return item.withText(normalizedLines.join(""));
  };
}
