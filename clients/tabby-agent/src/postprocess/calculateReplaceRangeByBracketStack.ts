import { CompletionContext, CompletionResponse } from "../CompletionContext";
import { isBlank, findUnpairedAutoClosingChars } from "../utils";
import { logger } from "./base";

export function calculateReplaceRangeByBracketStack(
  response: CompletionResponse,
  context: CompletionContext,
): CompletionResponse {
  const { currentLineSuffix } = context;
  const suffixText = currentLineSuffix.trimEnd();
  if (isBlank(suffixText)) {
    return response;
  }
  for (const choice of response.choices) {
    const completionText = choice.text.slice(context.position - choice.replaceRange.start);
    const unpaired = findUnpairedAutoClosingChars(completionText);
    if (isBlank(unpaired)) {
      continue;
    }
    if (suffixText.startsWith(unpaired)) {
      choice.replaceRange.end = context.position + unpaired.length;
      logger.trace(
        { context, completion: choice.text, range: choice.replaceRange, unpaired },
        "Adjust replace range by bracket stack",
      );
    } else if (unpaired.startsWith(suffixText)) {
      choice.replaceRange.end = context.position + suffixText.length;
      logger.trace(
        { context, completion: choice.text, range: choice.replaceRange, unpaired },
        "Adjust replace range by bracket stack",
      );
    }
  }
  return response;
}
