import { PostprocessFilter, logger } from "./base";
import { splitLines, isBlank, calcDistance } from "../utils";

export function removeRepetitiveLines(): PostprocessFilter {
  return (input: string) => {
    const inputLines = splitLines(input);
    let repetitionCount = 0;
    const repetitionThreshold = 5;
    // skip last line, it could be a not completed line
    let index = inputLines.length - 2;
    while (index >= 1) {
      if (isBlank(inputLines[index]!)) {
        index--;
        continue;
      }
      let prev = index - 1;
      while (prev >= 0 && isBlank(inputLines[prev]!)) {
        prev--;
      }
      if (prev < 0) break;
      // if distance between current and previous line is less than threshold (threshold = or 10% of string length)
      const currentLine = inputLines[index]!.trim();
      const previousLine = inputLines[prev]!.trim();
      const threshold = Math.max(0.1 * currentLine.length, 0.1 * previousLine.length);
      const distance = calcDistance(currentLine, previousLine);
      if (distance <= threshold) {
        repetitionCount++;
        index = prev;
      } else {
        break;
      }
    }
    if (repetitionCount >= repetitionThreshold) {
      logger.debug(
        {
          inputLines,
          repetitionCount,
        },
        "Remove repetitive lines.",
      );
      return inputLines
        .slice(0, index + 1)
        .join("")
        .trimEnd();
    }
    return input;
  };
}
