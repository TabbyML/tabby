import { getParser, languagesConfigs } from "../syntax/parser";
import { CompletionContext, CompletionResponse } from "../CompletionContext";
import { isBlank, splitLines } from "../utils";
import { logger } from "./base";

export const supportedLanguages = Object.keys(languagesConfigs);

export async function calculateReplaceRangeBySyntax(
  response: CompletionResponse,
  context: CompletionContext,
): Promise<CompletionResponse> {
  const { position, prefix, suffix, prefixLines, currentLineSuffix, currentLinePrefix, language } = context;
  if (!supportedLanguages.includes(language)) {
    return response;
  }
  const languageConfig = languagesConfigs[language]!;
  const parser = await getParser(languageConfig);
  const suffixText = currentLineSuffix.trimEnd();
  if (isBlank(suffixText)) {
    return response;
  }
  for (const choice of response.choices) {
    const completionText = choice.text.slice(position - choice.replaceRange.start);
    const completionLines = splitLines(completionText);
    let replaceLength = 0;
    let tree = parser.parse(prefix + completionText + suffix);
    let node = tree.rootNode.namedDescendantForIndex(prefix.length + completionText.length);
    while (node.hasError() && replaceLength < suffixText.length) {
      replaceLength++;
      const row = prefixLines.length - 1 + completionLines.length - 1;
      let column = completionLines[completionLines.length - 1]?.length ?? 0;
      if (completionLines.length == 1) {
        column += currentLinePrefix.length;
      }
      tree.edit({
        startIndex: prefix.length + completionText.length,
        oldEndIndex: prefix.length + completionText.length + 1,
        newEndIndex: prefix.length + completionText.length,
        startPosition: { row, column },
        oldEndPosition: { row, column: column + 1 },
        newEndPosition: { row, column },
      });
      tree = parser.parse(prefix + completionText + suffix.slice(replaceLength), tree);
      node = tree.rootNode.namedDescendantForIndex(prefix.length + completionText.length);
    }
    if (!node.hasError()) {
      choice.replaceRange.end = position + replaceLength;
      logger.trace({ context, completion: choice.text, range: choice.replaceRange }, "Adjust replace range by syntax");
    }
  }
  return response;
}
