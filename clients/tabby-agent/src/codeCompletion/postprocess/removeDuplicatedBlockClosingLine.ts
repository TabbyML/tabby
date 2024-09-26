import { PostprocessFilter, logger } from "./base";
import { CompletionItem } from "../solution";
import { isBlank, isBlockClosingLine } from "../../utils/string";

// For remove duplicated block closing line at ( ending of input text ) and ( beginning of suffix text )
// Should be useful after limitScope
export function removeDuplicatedBlockClosingLine(): PostprocessFilter {
  return (item: CompletionItem): CompletionItem => {
    const context = item.context;
    const { suffixLines, currentLinePrefix } = context;
    const inputLines = item.lines;
    if (inputLines.length < 2) {
      // If completion only has one line, don't continue process
      return item;
    }

    const inputLinesForDetection = inputLines.map((line, index) => {
      return index === 0 ? currentLinePrefix + line : line;
    });
    if (!isBlockClosingLine(inputLinesForDetection, inputLines.length - 1)) {
      return item;
    }
    const inputEndingLine = inputLines[inputLines.length - 1]!;

    let suffixBeginningIndex = 1;
    while (suffixBeginningIndex < suffixLines.length && isBlank(suffixLines[suffixBeginningIndex]!)) {
      suffixBeginningIndex++;
    }
    if (suffixBeginningIndex >= suffixLines.length) {
      return item;
    }
    const suffixBeginningLine = suffixLines[suffixBeginningIndex]!;

    if (
      inputEndingLine.startsWith(suffixBeginningLine.trimEnd()) ||
      suffixBeginningLine.startsWith(inputEndingLine.trimEnd())
    ) {
      logger.trace("Remove duplicated block closing line.", { inputLines, suffixLines });
      return item.withText(
        inputLines
          .slice(0, inputLines.length - 1)
          .join("")
          .trimEnd(),
      );
    }
    return item;
  };
}
