import { CompletionContext } from "../CompletionContext";
import { PostprocessFilter, logger } from "./base";
import { splitLines, isBlank, calcDistance } from "../utils";

export function dropDuplicated(): PostprocessFilter {
  return (input: string, context: CompletionContext) => {
    // get first n (n <= 3) lines of input and suffix, ignore blank lines
    const { suffixLines } = context;
    const inputLines = splitLines(input);
    let inputIndex = 0;
    while (inputIndex < inputLines.length && isBlank(inputLines[inputIndex]!)) {
      inputIndex++;
    }
    let suffixIndex = 0;
    while (suffixIndex < suffixLines.length && isBlank(suffixLines[suffixIndex]!)) {
      suffixIndex++;
    }
    const lineCount = Math.min(3, inputLines.length - inputIndex, suffixLines.length - suffixIndex);
    if (lineCount < 1) return input;
    const inputToCompare = inputLines
      .slice(inputIndex, inputIndex + lineCount)
      .join("")
      .trim();
    const suffixToCompare = suffixLines
      .slice(suffixIndex, suffixIndex + lineCount)
      .join("")
      .trim();
    // if string distance is less than threshold (threshold = 1, or 5% of string length)
    // drop this completion due to duplicated
    const threshold = Math.max(1, 0.05 * inputToCompare.length, 0.05 * suffixToCompare.length);
    const distance = calcDistance(inputToCompare, suffixToCompare);
    if (distance <= threshold) {
      logger.debug(
        { inputLines, suffixLines, inputToCompare, suffixToCompare, distance, threshold },
        "Drop completion due to duplicated.",
      );
      return null;
    }
    return input;
  };
}
