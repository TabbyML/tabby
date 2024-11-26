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

export function normalizeIndentation(): PostprocessFilter {
  return (item: CompletionItem): CompletionItem => {
    const { context, lines } = item;
    if (!Array.isArray(lines) || lines.length === 0) return item;

    // skip if current line has content
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
