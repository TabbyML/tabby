import { PostprocessFilter, logger } from "./base";
import { CompletionResultItem, emptyCompletionResultItem } from "../solution";
import { CompletionContext } from "../contexts";

export function trimMultiLineInSingleLineMode(): PostprocessFilter {
  return (item: CompletionResultItem, context: CompletionContext): CompletionResultItem => {
    const inputLines = item.lines;
    if (!context.isLineEnd && inputLines.length > 1) {
      const suffix = context.currentLineSuffix.trimEnd();
      const inputLine = inputLines[0]!.trimEnd();
      if (inputLine.endsWith(suffix)) {
        const trimmedInputLine = inputLine.slice(0, -suffix.length);
        if (trimmedInputLine.length > 0) {
          logger.trace("Trim content with multiple lines.", { inputLines, trimmedInputLine });
          return item.withText(trimmedInputLine);
        }
      }
      logger.trace("Drop content with multiple lines.", { inputLines });
      return emptyCompletionResultItem;
    }
    return item;
  };
}
