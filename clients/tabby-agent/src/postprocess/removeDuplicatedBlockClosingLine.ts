import { CompletionContext } from "../CompletionContext";
import { PostprocessFilter, logger } from "./base";
import { isBlank, splitLines, isBlockClosingLine } from "../utils";

// For remove duplicated block closing line at ( ending of input text ) and ( beginning of suffix text )
// Should be useful after limitScope
export function removeDuplicatedBlockClosingLine(): PostprocessFilter {
  return (input: string, context: CompletionContext) => {
    const { suffixLines, currentLinePrefix } = context;
    const inputLines = splitLines(input);
    if (inputLines.length < 2) {
      // If completion only has one line, don't continue process
      return input;
    }

    const inputLinesForDetection = inputLines.map((line, index) => {
      return index === 0 ? currentLinePrefix + line : line;
    });
    if (!isBlockClosingLine(inputLinesForDetection, inputLines.length - 1)) {
      return input;
    }
    const inputEndingLine = inputLines[inputLines.length - 1]!;

    let suffixBeginningIndex = 1;
    while (suffixBeginningIndex < suffixLines.length && isBlank(suffixLines[suffixBeginningIndex]!)) {
      suffixBeginningIndex++;
    }
    if (suffixBeginningIndex >= suffixLines.length) {
      return input;
    }
    const suffixBeginningLine = suffixLines[suffixBeginningIndex]!;

    if (inputEndingLine.startsWith(suffixBeginningLine) || suffixBeginningLine.startsWith(inputEndingLine)) {
      logger.debug({ inputLines, suffixLines }, "Removing duplicated block closing line");
      return inputLines
        .slice(0, inputLines.length - 1)
        .join("")
        .trimEnd();
    }
    return input;
  };
}
