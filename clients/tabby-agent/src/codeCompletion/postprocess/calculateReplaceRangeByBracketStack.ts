import { logger } from "./base";
import { CompletionItem } from "../solution";

export function calculateReplaceRangeByBracketStack(item: CompletionItem): CompletionItem {
  const context = item.context;
  const { currentLineSuffix } = context;

  if (!context.lineEnd) {
    return item;
  }

  let modified: CompletionItem | undefined = undefined;
  const suffixText = context.currentLineSuffix.trimEnd();
  const lineEnd = context.lineEnd[0];
  if (suffixText.startsWith(lineEnd)) {
    modified = item.withSuffix(lineEnd);
  } else if (lineEnd.startsWith(suffixText)) {
    modified = item.withSuffix(suffixText);
  }
  if (modified) {
    logger.trace("Adjust replace range by bracket stack.", {
      position: context.position,
      currentLineSuffix,
      completionText: item.text,
      lineEnd,
      replaceSuffix: item.replaceSuffix,
    });
    return modified;
  }
  return item;
}
