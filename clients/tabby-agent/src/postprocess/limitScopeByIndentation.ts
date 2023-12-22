import { CompletionContext } from "../CompletionContext";
import { AgentConfig } from "../AgentConfig";
import { PostprocessFilter, logger } from "./base";
import { isBlank, splitLines, getIndentationLevel, isBlockOpeningLine, isBlockClosingLine } from "../utils";

function parseIndentationContext(
  inputLines: string[],
  inputLinesForDetection: string[],
  context: CompletionContext,
  config: AgentConfig["postprocess"]["limitScope"]["indentation"],
): { indentLevelLimit: number; allowClosingLine: boolean } {
  const result = {
    indentLevelLimit: 0,
    allowClosingLine: true,
  };
  const { prefixLines, suffixLines, currentLinePrefix } = context;
  if (inputLines.length == 0 || prefixLines.length == 0) {
    return result; // guard for empty input, technically unreachable
  }
  const isCurrentLineInPrefixBlank = isBlank(currentLinePrefix);
  // if current line is blank, use the previous line as reference
  let referenceLineInPrefixIndex = prefixLines.length - 1;
  while (referenceLineInPrefixIndex >= 0 && isBlank(prefixLines[referenceLineInPrefixIndex]!)) {
    referenceLineInPrefixIndex--;
  }
  if (referenceLineInPrefixIndex < 0) {
    return result; // blank prefix, should be unreachable
  }
  const referenceLineInPrefix = prefixLines[referenceLineInPrefixIndex]!;
  const referenceLineInPrefixIndent = getIndentationLevel(referenceLineInPrefix);

  const currentLineInCompletion = inputLines[0]!;
  const isCurrentLineInCompletionBlank = isBlank(currentLineInCompletion);
  // if current line is blank, use the next line as reference
  let referenceLineInCompletionIndex = 0;
  while (referenceLineInCompletionIndex < inputLines.length && isBlank(inputLines[referenceLineInCompletionIndex]!)) {
    referenceLineInCompletionIndex++;
  }
  if (referenceLineInCompletionIndex >= inputLines.length) {
    return result; // blank completion, should be unreachable
  }
  const referenceLineInCompletion = inputLines[referenceLineInCompletionIndex]!;
  let referenceLineInCompletionIndent;
  if (isCurrentLineInCompletionBlank) {
    referenceLineInCompletionIndent = getIndentationLevel(referenceLineInCompletion);
  } else {
    referenceLineInCompletionIndent = getIndentationLevel(currentLinePrefix + referenceLineInCompletion);
  }

  if (!isCurrentLineInCompletionBlank && !isCurrentLineInPrefixBlank) {
    // if two reference lines are contacted at current line, it is continuing uncompleted sentence
    if (config.experimentalKeepBlockScopeWhenCompletingLine) {
      result.indentLevelLimit = referenceLineInPrefixIndent;
    } else {
      result.indentLevelLimit = referenceLineInPrefixIndent + 1; // + 1 for comparison, no matter how many spaces indent
      // allow closing line only if first line is opening a new indent block
      result.allowClosingLine &&= isBlockOpeningLine(inputLinesForDetection, 0);
    }
  } else if (referenceLineInCompletionIndent > referenceLineInPrefixIndent) {
    // if reference line in completion has more indent than reference line in prefix, it is opening a new indent block
    result.indentLevelLimit = referenceLineInPrefixIndent + 1;
  } else if (referenceLineInCompletionIndent < referenceLineInPrefixIndent) {
    // if reference line in completion has less indent than reference line in prefix, allow this closing
    result.indentLevelLimit = referenceLineInPrefixIndent;
  } else {
    // otherwise, it is starting a new sentence at same indent level
    result.indentLevelLimit = referenceLineInPrefixIndent;
  }

  // check if suffix context allows closing line
  // skip 0 that is current line in suffix
  let firstNonBlankLineInSuffix = 1;
  while (firstNonBlankLineInSuffix < suffixLines.length && isBlank(suffixLines[firstNonBlankLineInSuffix]!)) {
    firstNonBlankLineInSuffix++;
  }
  if (firstNonBlankLineInSuffix < suffixLines.length) {
    const firstNonBlankLineInSuffixText = suffixLines[firstNonBlankLineInSuffix]!;
    // allow closing line only if suffix has less indent level
    result.allowClosingLine &&= getIndentationLevel(firstNonBlankLineInSuffixText) < result.indentLevelLimit;
  }
  return result;
}

export function limitScopeByIndentation(
  config: AgentConfig["postprocess"]["limitScope"]["indentation"],
): PostprocessFilter {
  return (input: string, context: CompletionContext) => {
    const { prefixLines, suffixLines, currentLinePrefix } = context;
    const inputLines = splitLines(input);
    const inputLinesForDetection = inputLines.map((line, index) => {
      return index === 0 ? currentLinePrefix + line : line;
    });
    const indentContext = parseIndentationContext(inputLines, inputLinesForDetection, context, config);
    let index = 1;
    while (index < inputLines.length) {
      const line = inputLines[index]!;
      const prevLine = inputLines[index - 1]!;
      if (isBlank(line)) {
        index++;
        continue;
      }
      const indentLevel = getIndentationLevel(line);
      // If the line is indented less than the indent level limit, it is closing indent block.
      if (indentLevel < indentContext.indentLevelLimit) {
        // But when it is also opening a new indent block immediately, such as `} else {`, continue.
        if (isBlockClosingLine(inputLinesForDetection, index) && isBlockOpeningLine(inputLinesForDetection, index)) {
          index++;
          continue;
        }
        // If context allows, we should add the block closing line
        // For python, if previous line is blank, we don't include this line
        if (indentContext.allowClosingLine && (context.language !== "python" || !isBlank(prevLine))) {
          index++;
        }
        break;
      }
      // else continue
      index++;
    }
    if (index < inputLines.length) {
      logger.debug(
        { inputLines, prefixLines, suffixLines, scopeLineCount: index },
        "Remove content out of indent scope",
      );
      return inputLines.slice(0, index).join("").trimEnd();
    }
    return input;
  };
}
