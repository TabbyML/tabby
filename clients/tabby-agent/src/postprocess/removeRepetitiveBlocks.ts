import { CompletionContext } from "../CompletionContext";
import { PostprocessFilter, logger } from "./base";
import { isBlank, calcDistance } from "../utils";

function blockSplitter(_: string) {
  // Have not implemented this for each language for now
  // Return a blank line matcher should work for most cases
  return /\n(\s*)\n/g;
}

// FIXME: refactor this because it is very similar to `removeRepetitiveLines`
export function removeRepetitiveBlocks(): PostprocessFilter {
  return (input: string, context: CompletionContext) => {
    const inputBlocks = input.split(blockSplitter(context.language));
    let repetitionCount = 0;
    const repetitionThreshold = 2;
    // skip last block, it maybe cut
    let index = inputBlocks.length - 2;
    while (index >= 1) {
      if (isBlank(inputBlocks[index]!)) {
        index--;
        continue;
      }
      let prev = index - 1;
      while (prev >= 0 && isBlank(inputBlocks[prev]!)) {
        prev--;
      }
      if (prev < 0) break;
      // if distance between current and previous block is less than threshold (threshold = or 10% of string length)
      const currentBlock = inputBlocks[index]!.trim();
      const previousBlock = inputBlocks[prev]!.trim();
      const threshold = Math.max(0.1 * currentBlock.length, 0.1 * previousBlock.length);
      const distance = calcDistance(currentBlock, previousBlock);
      if (distance <= threshold) {
        repetitionCount++;
        index--;
      } else {
        break;
      }
    }
    if (repetitionCount >= repetitionThreshold) {
      logger.debug(
        {
          inputBlocks,
          repetitionCount,
        },
        "Remove repetitive blocks.",
      );
      return inputBlocks
        .slice(0, index + 1)
        .join("")
        .trimEnd();
    }
    return input;
  };
}
