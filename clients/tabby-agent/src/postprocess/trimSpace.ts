import { CompletionContext } from "../Agent";
import { PostprocessFilter, logger } from "./base";
import { splitLines, isBlank } from "../utils";

export const trimSpace: (context: CompletionContext) => PostprocessFilter = (context) => {
  return (input) => {
    const { prefixLines, suffixLines } = context;
    const inputLines = splitLines(input);
    let trimmedInput = input;
    const prefixCurrentLine = prefixLines[prefixLines.length - 1] ?? "";
    const suffixCurrentLine = suffixLines[0] ?? "";
    if (!isBlank(prefixCurrentLine) && prefixCurrentLine.match(/\s$/)) {
      trimmedInput = trimmedInput.trimStart();
    }

    if (
      inputLines.length > 1 ||
      isBlank(suffixCurrentLine) ||
      (!isBlank(suffixCurrentLine) && suffixCurrentLine.match(/^\s/))
    ) {
      trimmedInput = trimmedInput.trimEnd();
    }
    return trimmedInput;
  };
};
