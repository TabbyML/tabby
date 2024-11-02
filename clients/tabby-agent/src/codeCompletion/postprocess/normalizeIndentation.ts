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
      const cursorLineSpaces = getLeadingSpaces(context.currentLinePrefix);
      const firstLineSpaces = getLeadingSpaces(firstLine);

      // check if any following line matches cursor indentation
      let hasMatchingIndent = false;
      for (let i = 1; i < lines.length; i++) {
        const line = lines[i];
        if (line && line.trim() && getLeadingSpaces(line) === cursorLineSpaces) {
          hasMatchingIndent = true;
          break;
        }
      }

      // if cursor line has even spaces, process indentation
      if (cursorLineSpaces % 2 === 0 && firstLineSpaces > 0) {
        if (hasMatchingIndent) {
          // if matching indent found, only process first line
          const lineSpaces = getLeadingSpaces(firstLine);
          const spacesToRemove = Math.min(lineSpaces, firstLineSpaces);
          normalizedLines[0] = firstLine.substring(spacesToRemove);
        } else {
          // otherwise process all lines
          normalizedLines.forEach((line, index) => {
            if (!line || !line.trim()) return; // skip empty lines
            const lineSpaces = getLeadingSpaces(line);
            const spacesToRemove = Math.min(lineSpaces, firstLineSpaces);
            normalizedLines[index] = line.substring(spacesToRemove);
          });
        }
      }
    }

    return item.withText(normalizedLines.join(""));
  };
}
