import { isBlank } from "../../utils/string";
import { CompletionItem } from "../solution";
import { logger } from "./base";

export function calculateReplaceRangeBySemiColon(item: CompletionItem): CompletionItem {
  const context = item.context;
  const { currentLineSuffix } = context;
  const suffixText = currentLineSuffix.trimEnd();

  if (isBlank(suffixText)) {
    return item;
  }

  const completionText = item.text.trimEnd();
  if (!completionText.includes(";")) {
    return item;
  }

  if (suffixText.startsWith(";") && completionText.endsWith(";")) {
    const modified = item.withSuffix(";");

    logger.trace("Adjust replace range by semicolon.", {
      position: context.position,
      currentLineSuffix,
      completionText: item.text,
      modifiedText: item.text,
      replaceSuffix: modified.replaceSuffix,
    });

    return modified;
  }

  return item;
}
