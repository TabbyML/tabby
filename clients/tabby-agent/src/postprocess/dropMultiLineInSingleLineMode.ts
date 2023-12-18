import { CompletionContext } from "../CompletionContext";
import { PostprocessFilter, logger } from "./base";
import { splitLines } from "../utils";

export function dropMultiLineInSingleLineMode(): PostprocessFilter {
  return (input: string, context: CompletionContext) => {
    const inputLines = splitLines(input);
    if (context.mode === "fill-in-line" && inputLines.length > 1) {
      logger.debug({ inputLines }, "Drop content with multiple lines");
      return null;
    }
    return input;
  };
}
