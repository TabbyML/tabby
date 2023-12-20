import { CompletionContext } from "../CompletionContext";
import { AgentConfig } from "../AgentConfig";
import { PostprocessFilter, logger } from "./base";
import { isBlank, splitLines } from "../utils";

function calcIndentLevel(line: string): number {
  return line.match(/^[ \t]*/)?.[0]?.length ?? 0;
}

function isOpeningIndentBlock(lines: string[], index: number): boolean {
  if (index < 0 || index >= lines.length - 1) {
    return false;
  }
  return calcIndentLevel(lines[index]!) < calcIndentLevel(lines[index + 1]!);
}

function processContext(
  lines: string[],
  context: CompletionContext,
  config: AgentConfig["postprocess"]["limitScope"]["indentation"],
): { indentLevelLimit: number; allowClosingLine: (closingLine: string) => boolean } {
  let allowClosingLine = false;
  const result = { indentLevelLimit: 0, allowClosingLine: (_: string) => allowClosingLine };
  const { prefixLines, suffixLines, currentLinePrefix } = context;
  if (lines.length == 0 || prefixLines.length == 0) {
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
  const referenceLineInPrefixIndent = calcIndentLevel(referenceLineInPrefix);

  const currentLineInCompletion = lines[0]!;
  const isCurrentLineInCompletionBlank = isBlank(currentLineInCompletion);
  // if current line is blank, use the next line as reference
  let referenceLineInCompletionIndex = 0;
  while (referenceLineInCompletionIndex < lines.length && isBlank(lines[referenceLineInCompletionIndex]!)) {
    referenceLineInCompletionIndex++;
  }
  if (referenceLineInCompletionIndex >= lines.length) {
    return result; // blank completion, should be unreachable
  }
  const referenceLineInCompletion = lines[referenceLineInCompletionIndex]!;
  let referenceLineInCompletionIndent;
  if (isCurrentLineInCompletionBlank) {
    referenceLineInCompletionIndent = calcIndentLevel(referenceLineInCompletion);
  } else {
    referenceLineInCompletionIndent = calcIndentLevel(currentLinePrefix + referenceLineInCompletion);
  }

  if (!isCurrentLineInCompletionBlank && !isCurrentLineInPrefixBlank) {
    // if two reference lines are contacted at current line, it is continuing uncompleted sentence

    if (config.experimentalKeepBlockScopeWhenCompletingLine) {
      result.indentLevelLimit = referenceLineInPrefixIndent;
    } else {
      result.indentLevelLimit = referenceLineInPrefixIndent + 1; // + 1 for comparison, no matter how many spaces indent
    }
    // allow closing line if first line is opening a new indent block
    allowClosingLine = !!lines[1] && calcIndentLevel(lines[1]) > referenceLineInPrefixIndent;
  } else if (referenceLineInCompletionIndent > referenceLineInPrefixIndent) {
    // if reference line in completion has more indent than reference line in prefix, it is opening a new indent block

    result.indentLevelLimit = referenceLineInPrefixIndent + 1;
    allowClosingLine = true;
  } else if (referenceLineInCompletionIndent < referenceLineInPrefixIndent) {
    // if reference line in completion has less indent than reference line in prefix, allow this closing

    result.indentLevelLimit = referenceLineInPrefixIndent;
    allowClosingLine = true;
  } else {
    // otherwise, it is starting a new sentence at same indent level

    result.indentLevelLimit = referenceLineInPrefixIndent;
    allowClosingLine = true;
  }

  // check if suffix context allows closing line
  // skip 0 that is current line in suffix
  let firstNonBlankLineInSuffix = 1;
  while (firstNonBlankLineInSuffix < suffixLines.length && isBlank(suffixLines[firstNonBlankLineInSuffix]!)) {
    firstNonBlankLineInSuffix++;
  }
  if (firstNonBlankLineInSuffix < suffixLines.length) {
    const firstNonBlankLineInSuffixText = suffixLines[firstNonBlankLineInSuffix]!;
    allowClosingLine &&= calcIndentLevel(firstNonBlankLineInSuffixText) < result.indentLevelLimit;
    result.allowClosingLine = (closingLine: string) => {
      const duplicatedClosingLine =
        closingLine.startsWith(firstNonBlankLineInSuffixText) || firstNonBlankLineInSuffixText.startsWith(closingLine);
      return allowClosingLine && !duplicatedClosingLine;
    };
  }
  return result;
}

export function limitScopeByIndentation(
  config: AgentConfig["postprocess"]["limitScope"]["indentation"],
): PostprocessFilter {
  return (input: string, context: CompletionContext) => {
    const { prefixLines, suffixLines } = context;
    const inputLines = splitLines(input);
    const indentContext = processContext(inputLines, context, config);
    let index;
    for (index = 1; index < inputLines.length; index++) {
      const line = inputLines[index]!;
      const prevLine = inputLines[index - 1]!;
      if (isBlank(line)) {
        continue;
      }
      const indentLevel = calcIndentLevel(line);
      if (indentLevel < indentContext.indentLevelLimit) {
        // If the line is indented less than the indent level limit, it is closing indent block.
        // But when it is opening a new indent block immediately, such as `} else {`.
        if (isOpeningIndentBlock(inputLines, index)) {
          continue;
        }
        // We include this closing line here if context allows
        // For python, if previous line is blank, we don't include this line
        if (indentContext.allowClosingLine(line) && (context.language !== "python" || !isBlank(prevLine))) {
          index++;
        }
        break;
      }
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
