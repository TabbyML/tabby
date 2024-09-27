import { PostprocessFilter, logger } from "./base";
import { CompletionItem } from "../solution";

export function trimMultiLineInSingleLineMode(): PostprocessFilter {
  return (item: CompletionItem): CompletionItem => {
    const context = item.context;
    const inputLines = item.lines;
    if (context.mode === "fill-in-line" && inputLines.length > 1) {
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
      return CompletionItem.createBlankItem(context);
    }
    return item;
  };
}
