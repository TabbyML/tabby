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
    if (firstLine) {
      const firstLineSpaces = getLeadingSpaces(firstLine);
      const cursorLineSpaces = getLeadingSpaces(context.currentLinePrefix);
      console.log("firstLineSpaces", firstLineSpaces);
      console.log("curr prefix", context.currentLinePrefix);
      console.log("cursorLineSpaces", cursorLineSpaces);
      // deal with current cursor odd indentation
      if ((firstLineSpaces + cursorLineSpaces) % 2 !== 0 && firstLineSpaces % 2 !== 0) {
        normalizedLines[0] = firstLine.substring(firstLineSpaces);
        console.log("after normalize:", normalizedLines[0]);
      }
    }

    //deal with extra space in the line indent
    let indents = -1; // use track previous line indentations
    for (let i = 0; i < normalizedLines.length; i++) {
      const line = normalizedLines[i];
      if (!line || !line.trim()) continue;
      const lineSpaces = getLeadingSpaces(line);
      if (indents != -1 && lineSpaces % 2 !== 0) {
        // move current line to recently close indent
        normalizedLines[i] = " ".repeat(indents) + line.trimStart();
      } else {
        indents = lineSpaces;
      }
    }

    return item.withText(normalizedLines.join(""));
  };
}
