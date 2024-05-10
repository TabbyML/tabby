import { CompletionContext, CompletionResponseChoice } from "../CompletionContext";
import { isBlank, findUnpairedAutoClosingChars } from "../utils";
import { logger } from "./base";

export function calculateReplaceRangeByBracketStack(
  choice: CompletionResponseChoice,
  context: CompletionContext,
): CompletionResponseChoice {
  const { currentLineSuffix } = context;
  const suffixText = currentLineSuffix.trimEnd();
  if (isBlank(suffixText)) {
    return choice;
  }
  const completionText = choice.text.slice(context.position - choice.replaceRange.start);
  const unpaired = findUnpairedAutoClosingChars(completionText).join("");
  if (isBlank(unpaired)) {
    return choice;
  }
  if (suffixText.startsWith(unpaired)) {
    choice.replaceRange.end = context.position + unpaired.length;
  } else if (unpaired.startsWith(suffixText)) {
    choice.replaceRange.end = context.position + suffixText.length;
  }
  logger.trace("Adjust replace range by bracket stack.", {
    position: context.position,
    currentLineSuffix,
    completionText: choice.text,
    unpaired,
    replaceRange: choice.replaceRange,
  });
  return choice;
}
