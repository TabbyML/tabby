import { PostprocessFilter, PostprocessContext, logger } from "./filter";
import { isBlank, splitLines } from "../utils";

function calcIndentLevel(line) {
  return line.match(/^[ \t]*/)?.[0]?.length || 0;
}

function isIndentBlockClosingAllowed(currentIndentLevel, suffixLines) {
  let index = 1;
  while (index < suffixLines.length && isBlank(suffixLines[index])) {
    index++;
  }
  if (index >= suffixLines.length) {
    return true;
  } else {
    const indentLevel = calcIndentLevel(suffixLines[index]);
    return indentLevel < currentIndentLevel;
  }
}

function isOpeningIndentBlock(lines, index) {
  if (index >= lines.length - 1) {
    return false;
  }
  return calcIndentLevel(lines[index]) < calcIndentLevel(lines[index + 1]);
}

export const limitScopeByIndentation: (context: PostprocessContext) => PostprocessFilter = (context) => {
  return (input) => {
    const prefix = context.text.slice(0, context.position);
    const suffix = context.text.slice(context.position);
    const prefixLines = splitLines(prefix);
    const suffixLines = splitLines(suffix);
    const inputLines = splitLines(input);
    const currentIndentLevel = calcIndentLevel(prefixLines[prefixLines.length - 1]);
    let index;
    for (index = 1; index < inputLines.length; index++) {
      if (isBlank(inputLines[index])) {
        continue;
      }
      const indentLevel = calcIndentLevel(inputLines[index]);
      if (indentLevel < currentIndentLevel) {
        // If the line is indented less than the current indent level, it is out of scope.
        // We assume it begins with a symbol closing block.
        // If suffix context allows, and it do not open a new intent block, include this line here.
        if (isIndentBlockClosingAllowed(currentIndentLevel, suffixLines) && !isOpeningIndentBlock(inputLines, index)) {
          index++;
        }
        break;
      }
    }
    return inputLines.slice(0, index).join("").trimEnd();
  };
};
