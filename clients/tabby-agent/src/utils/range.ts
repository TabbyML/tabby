import { Position, Range } from "vscode-languageserver";
import { TextDocument } from "vscode-languageserver-textdocument";

export function isPositionEqual(a: Position, b: Position): boolean {
  return a.line === b.line && a.character === b.character;
}
export function isPositionBefore(a: Position, b: Position): boolean {
  return a.line < b.line || (a.line === b.line && a.character < b.character);
}
export function isPositionAfter(a: Position, b: Position): boolean {
  return a.line > b.line || (a.line === b.line && a.character > b.character);
}
export function isPositionBeforeOrEqual(a: Position, b: Position): boolean {
  return a.line < b.line || (a.line === b.line && a.character <= b.character);
}
export function isPositionAfterOrEqual(a: Position, b: Position): boolean {
  return a.line > b.line || (a.line === b.line && a.character >= b.character);
}
export function isPositionInRange(a: Position, b: Range): boolean {
  return isPositionBeforeOrEqual(b.start, a) && isPositionBeforeOrEqual(a, b.end);
}
export function isRangeEqual(a: Range, b: Range): boolean {
  return isPositionEqual(a.start, b.start) && isPositionEqual(a.end, b.end);
}
export function isEmptyRange(a: Range): boolean {
  return isPositionAfterOrEqual(a.start, a.end);
}
export function unionRange(a: Range, b: Range): Range {
  return {
    start: isPositionBefore(a.start, b.start)
      ? { line: a.start.line, character: a.start.character }
      : { line: b.start.line, character: b.start.character },
    end: isPositionAfter(a.end, b.end)
      ? { line: a.end.line, character: a.end.character }
      : { line: b.end.line, character: b.end.character },
  };
}
export function intersectionRange(a: Range, b: Range): Range | null {
  const range = {
    start: isPositionAfter(a.start, b.start)
      ? { line: a.start.line, character: a.start.character }
      : { line: b.start.line, character: b.start.character },
    end: isPositionBefore(a.end, b.end)
      ? { line: a.end.line, character: a.end.character }
      : { line: b.end.line, character: b.end.character },
  };
  return isEmptyRange(range) ? null : range;
}
export function documentRange(doc: TextDocument): Range {
  return {
    start: {
      line: 0,
      character: 0,
    },
    end: {
      line: doc.lineCount,
      character: 0,
    },
  };
}
export function rangeInDocument(a: Range, doc: TextDocument): Range | null {
  return intersectionRange(a, documentRange(doc));
}
