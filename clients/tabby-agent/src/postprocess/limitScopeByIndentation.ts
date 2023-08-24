import { PostprocessFilter, PostprocessContext, logger } from "./base";
import { isBlank, splitLines } from "../utils";

function calcIndentLevel(line: string): number {
  return line.match(/^[ \t]*/)?.[0]?.length ?? 0;
}

function isOpeningIndentBlock(lines, index) {
  if (index >= lines.length - 1) {
    return false;
  }
  return calcIndentLevel(lines[index]) < calcIndentLevel(lines[index + 1]);
}

function shouldOnlyAllowSingleLine(suffixLines: string[]): boolean {
  let currentLineInSuffix = suffixLines[0] ?? "";
  return !isBlank(currentLineInSuffix.replace(/[\)\}\]]/g, ""));
}

function processContext(
  lines: string[],
  prefixLines: string[],
  suffixLines: string[],
): { indentLevelLimit: number; allowClosingLine: boolean } {
  let result = { indentLevelLimit: 0, allowClosingLine: false };
  if (lines.length == 0 || prefixLines.length == 0) {
    return result; // guard for empty input, technically unreachable
  }
  const currentLineInPrefix = prefixLines[prefixLines.length - 1];
  const isCurrentLineInPrefixBlank = isBlank(currentLineInPrefix);
  // if current line is blank, use the previous line as reference
  let referenceLineInPrefixIndex = prefixLines.length - 1;
  while (referenceLineInPrefixIndex >= 0 && isBlank(prefixLines[referenceLineInPrefixIndex])) {
    referenceLineInPrefixIndex--;
  }
  if (referenceLineInPrefixIndex < 0) {
    return result; // blank prefix, should be unreachable
  }
  const referenceLineInPrefix = prefixLines[referenceLineInPrefixIndex];
  const referenceLineInPrefixIndent = calcIndentLevel(referenceLineInPrefix);

  const currentLineInCompletion = lines[0];
  const isCurrentLineInCompletionBlank = isBlank(currentLineInCompletion);
  // if current line is blank, use the next line as reference
  let referenceLineInCompletionIndex = 0;
  while (referenceLineInCompletionIndex < lines.length && isBlank(lines[referenceLineInCompletionIndex])) {
    referenceLineInCompletionIndex++;
  }
  if (referenceLineInCompletionIndex >= lines.length) {
    return result; // blank completion, should be unreachable
  }
  const referenceLineInCompletion = lines[referenceLineInCompletionIndex];
  let referenceLineInCompletionIndent;
  if (isCurrentLineInCompletionBlank) {
    referenceLineInCompletionIndent = calcIndentLevel(referenceLineInCompletion);
  } else {
    referenceLineInCompletionIndent = calcIndentLevel(currentLineInPrefix + referenceLineInCompletion);
  }

  if (!isCurrentLineInCompletionBlank && !isCurrentLineInPrefixBlank) {
    // if two reference lines are contacted at current line, it is continuing uncompleted sentence

    result.indentLevelLimit = referenceLineInPrefixIndent + 1; // + 1 for comparison, no matter how many spaces indent
    // allow closing line if first line is opening a new indent block
    result.allowClosingLine = !!lines[1] && calcIndentLevel(lines[1]) > referenceLineInPrefixIndent;
  } else if (referenceLineInCompletionIndent > referenceLineInPrefixIndent) {
    // if reference line in completion has more indent than reference line in prefix, it is opening a new indent block

    result.indentLevelLimit = referenceLineInPrefixIndent + 1;
    result.allowClosingLine = true;
  } else if (referenceLineInCompletionIndent < referenceLineInPrefixIndent) {
    // if reference line in completion has less indent than reference line in prefix, allow this closing

    result.indentLevelLimit = referenceLineInPrefixIndent;
    result.allowClosingLine = true;
  } else {
    // otherwise, it is starting a new sentence at same indent level

    result.indentLevelLimit = referenceLineInPrefixIndent;
    result.allowClosingLine = false;
  }

  // check if suffix context allows closing line
  // skip 0 that is current line in suffix, it is processed in `shouldOnlyAllowSingleLine`
  let firstNonBlankLineInSuffix = 1;
  while (firstNonBlankLineInSuffix < suffixLines.length && isBlank(suffixLines[firstNonBlankLineInSuffix])) {
    firstNonBlankLineInSuffix++;
  }
  if (firstNonBlankLineInSuffix < suffixLines.length) {
    result.allowClosingLine &&= calcIndentLevel(suffixLines[firstNonBlankLineInSuffix]) < result.indentLevelLimit;
  }
  return result;
}

export const limitScopeByIndentation: (context: PostprocessContext) => PostprocessFilter = (context) => {
  return (input) => {
    const { prefix, suffix, prefixLines, suffixLines } = context;
    const inputLines = splitLines(input);
    if (shouldOnlyAllowSingleLine(suffixLines)) {
      if (inputLines.length > 1) {
        logger.debug({ input, prefix, suffix }, "Drop content with multiple lines");
        return null;
      }
    }
    const indentContext = processContext(inputLines, prefixLines, suffixLines);
    let index;
    for (index = 1; index < inputLines.length; index++) {
      if (isBlank(inputLines[index])) {
        continue;
      }
      const indentLevel = calcIndentLevel(inputLines[index]);
      if (indentLevel < indentContext.indentLevelLimit) {
        // If the line is indented less than the indent level limit, it is closing indent block.
        // But when it is opening a new indent block immediately, such as `} else {`.
        if (isOpeningIndentBlock(inputLines, index)) {
          continue;
        }
        // We include this closing line here if context allows
        // Python does not have closing bracket, so we always include closing line
        if (indentContext.allowClosingLine && context.request.language !== "python") {
          index++;
        }
        break;
      }
    }
    if (index < inputLines.length) {
      logger.debug({ input, prefix, suffix, scopeEndAt: index }, "Remove content out of scope");
      return inputLines.slice(0, index).join("").trimEnd();
    }
    return input;
  };
};
