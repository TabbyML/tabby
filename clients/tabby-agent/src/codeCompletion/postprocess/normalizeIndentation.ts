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
 * 1. Early returns: The function returns the original item without modification if any of these conditions are met:
 *    - Lines array is invalid or empty
 *    - Current line prefix is empty
 *    - Current line prefix contains a tab character
 *
 * 2. Space-based indentation check:
 *    - If the current line prefix contains any non-space characters, return the original item
 *    - Otherwise, proceed with indentation normalization
 *
 * 3. Indentation normalization process:
 *    - For the first line: Calculate total leading spaces (cursor prefix + first line spaces)
 *      If both the total and the first line spaces are odd, remove all leading spaces from first line
 *      This prevents misalignment in cases of odd indentation units
 *
 * 4. Line-by-line normalization:
 *    - For each non-empty line: if leading spaces count is odd (and > 0),
 *      reduce it by 1 space to maintain even indentation
 *    - Empty or whitespace-only lines are skipped
 *
 * The adjustments ensure consistent indentation alignment throughout the code snippet,
 * particularly focusing on maintaining even space counts for proper alignment.
 */
export function normalizeIndentation(): PostprocessFilter {
  return (item: CompletionItem): CompletionItem => {
    const { context, lines } = item;
    if (
      !Array.isArray(lines) ||
      lines.length === 0 ||
      context.currentLinePrefix.length == 0 ||
      context.currentLinePrefix.includes("\t")
    )
      return item;

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
