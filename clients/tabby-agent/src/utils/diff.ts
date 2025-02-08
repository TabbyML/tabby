import { Range, Position } from "vscode-languageserver";
import { linesDiffComputers, Range as DiffRange } from "codiff";

interface CodeDiffResult {
  originRanges: Range[];
  modifiedRanges: Range[];
}

export function mapDiffRangeToEditorRange(diffRange: DiffRange, editorRanges: Range[]): Range | undefined {
  if (diffRange.isEmpty()) {
    return undefined;
  }

  const start: Position = {
    line: editorRanges[diffRange.startLineNumber - 1]?.start.line ?? 0,
    character: diffRange.startColumn - 1,
  };

  let end: Position;

  /**
   * In most case, start line and end line are equal in diff change.
   * when start line and end line are different in change, it usually means a range that include a whole line.
   * {
   *   "startLineNumber": 2,
   *   "startColumn": 1,
   *   "endLineNumber": 3,
   *   "endColumn": 1
   * }
   *
   * In our case, the origin code and modified code are mixed tegether. so we should translate range to below to avoid wrong range mapping.
   * {
   *   "startLineNumber": 2,
   *   "startColumn": 1,
   *   "endLineNumber": 2,
   *   "endColumn": // end of line 2
   * }
   *
   */
  if (diffRange.isSingleLine()) {
    end = {
      line: editorRanges[diffRange.startLineNumber - 1]?.start.line ?? 0,
      character: diffRange.endColumn - 1,
    };
  } else {
    end = {
      line: editorRanges[diffRange.startLineNumber - 1]?.start.line ?? 0,
      character: editorRanges[diffRange.startLineNumber - 1]?.end.character ?? 0,
    };
  }

  return {
    start,
    end,
  };
}

/**
 * Diff code and mapping the diff result range to editor range
 */
export function codeDiff(
  originCode: string[],
  originCodeRanges: Range[],
  modifiedCode: string[],
  modifiedCodeRanges: Range[],
): CodeDiffResult {
  const originRanges: Range[] = [];
  const modifiedRanges: Range[] = [];

  const diffResult = linesDiffComputers.getDefault().computeDiff(originCode, modifiedCode, {
    computeMoves: false,
    ignoreTrimWhitespace: true,
    maxComputationTimeMs: 100,
  });

  diffResult.changes.forEach((change) => {
    change.innerChanges?.forEach((innerChange) => {
      const originRange = mapDiffRangeToEditorRange(innerChange.originalRange, originCodeRanges);
      if (originRange) {
        originRanges.push(originRange);
      }

      const modifiedRange = mapDiffRangeToEditorRange(innerChange.modifiedRange, modifiedCodeRanges);
      if (modifiedRange) {
        modifiedRanges.push(modifiedRange);
      }
    });
  });

  return {
    modifiedRanges,
    originRanges,
  };
}
