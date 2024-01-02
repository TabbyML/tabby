import { getParser, languagesConfigs } from "../syntax/parser";
import { CompletionContext, CompletionResponseChoice } from "../CompletionContext";
import { isBlank, splitLines } from "../utils";
import { logger } from "./base";

export const supportedLanguages = Object.keys(languagesConfigs);

/**
 * @throws {Error} if language is not supported
 * @throws {Error} if syntax error when parsing completion
 */
export async function calculateReplaceRangeBySyntax(
  choice: CompletionResponseChoice,
  context: CompletionContext,
): Promise<CompletionResponseChoice> {
  const { position, prefix, suffix, prefixLines, currentLinePrefix, currentLineSuffix, language } = context;
  const suffixText = currentLineSuffix.trimEnd();
  if (isBlank(suffixText)) {
    return choice;
  }

  if (!supportedLanguages.includes(language)) {
    throw new Error(`Language ${language} is not supported`);
  }
  const languageConfig = languagesConfigs[language]!;
  const parser = await getParser(languageConfig);

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
  if (node.hasError()) {
    throw new Error("Syntax error when parsing completion");
  }

  choice.replaceRange.end = position + replaceLength;
  logger.trace({ context, completion: choice.text, range: choice.replaceRange }, "Adjust replace range by syntax");
  return choice;
}
