import { Position, Range } from "vscode-languageserver";
import { diffChars } from "diff";

export function calcCharDiffRange(originText: string, editText: string, editTextRanges: Range[]): Range[] {
  const diffRanges: Range[] = [];
  const changes = diffChars(originText, editText);
  let index = 0;
  changes.forEach((item) => {
    if (item.added) {
      const position = getPositionFromIndex(index, editTextRanges);
      const addedRange: Range = {
        start: position,
        end: { line: position.line, character: position.character + (item.count ?? 0) },
      };
      diffRanges.push(addedRange);
      index += item.count ?? 0;
    } else if (item.removed) {
      // nothing
    } else {
      index += item.count ?? 0;
    }
  });
  return diffRanges;
}

export function getPositionFromIndex(index: number, ranges: Range[]): Position {
  let line = 0;
  let character = 0;
  let length = 0;
  for (let i = 0; i < ranges.length; i++) {
    const range = ranges[i];
    if (!range) {
      continue;
    }
    const rangeLength = range.end.character - range.start.character + length + 1;
    if (index >= length && index < rangeLength) {
      line = range.start.line;
      character = index - length;
      return {
        line,
        character,
      };
    } else {
      length = rangeLength;
    }
  }

  return {
    line,
    character,
  };
}
