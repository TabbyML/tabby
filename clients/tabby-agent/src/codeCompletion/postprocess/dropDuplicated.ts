import { PostprocessFilter, logger } from "./base";
import { CompletionItem } from "../solution";
import { isBlank, calcDistance } from "../../utils/string";

export function dropDuplicated(): PostprocessFilter {
  return (item: CompletionItem): CompletionItem => {
    const context = item.context;
    // get first n (n <= 3) lines of input and suffix, ignore blank lines
    const { suffixLines } = context;
    const inputLines = item.lines;
    let inputIndex = 0;
    while (inputIndex < inputLines.length && isBlank(inputLines[inputIndex]!)) {
      inputIndex++;
    }
    let suffixIndex = 0;
    while (suffixIndex < suffixLines.length && isBlank(suffixLines[suffixIndex]!)) {
      suffixIndex++;
    }
    const lineCount = Math.min(3, inputLines.length - inputIndex, suffixLines.length - suffixIndex);
    if (lineCount < 1) {
      return item;
    }
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
      logger.trace("Drop completion due to duplicated.", { inputToCompare, suffixToCompare, distance, threshold });
      return CompletionItem.createBlankItem(context);
    }
    return item;
  };
}
