import { logger } from "./base";
import { CompletionItem } from "../solution";
import { isBlank, findUnpairedAutoClosingChars } from "../../utils/string";

export function calculateReplaceRangeByBracketStack(item: CompletionItem): CompletionItem {
  const context = item.context;
  const { currentLineSuffix } = context;
  const suffixText = currentLineSuffix.trimEnd();
  if (isBlank(suffixText)) {
    return item;
  }
  const unpaired = findUnpairedAutoClosingChars(item.text).join("");
  if (isBlank(unpaired)) {
    return item;
  }

  let modified: CompletionItem | undefined = undefined;
  if (suffixText.startsWith(unpaired)) {
    modified = item.withSuffix(unpaired);
  } else if (unpaired.startsWith(suffixText)) {
    modified = item.withSuffix(suffixText);
  }
  if (modified) {
    logger.trace("Adjust replace range by bracket stack.", {
      position: context.position,
      currentLineSuffix,
      completionText: item.text,
      unpaired,
      replaceSuffix: item.replaceSuffix,
    });
    return modified;
  }
  return item;
}
