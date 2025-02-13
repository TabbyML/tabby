import { Range, Position } from "vscode-languageserver";
import { linesDiffComputers, Range as DiffRange } from "codiff";

interface CodeDiffResult {
  originRanges: Range[];
  modifiedRanges: Range[];
}

function splitRangeToSingleLine(range: DiffRange, codeLines: string[]): DiffRange[] {
  if (range.isSingleLine()) {
    return [range];
  }
  const resultRanges: DiffRange[] = [];
  for (let i = range.startLineNumber; i <= range.endLineNumber; i++) {
    const singlelineRange = new DiffRange(
      i,
      i === range.startLineNumber ? range.startColumn : 1,
      i,
      i === range.endLineNumber ? range.endColumn : codeLines[i - 1]?.length ?? 1,
    );
    resultRanges.push(singlelineRange);
  }
  return resultRanges;
}

function mapDiffRangeToEditorRange(diffRange: DiffRange, editorRanges: Range[]): Range | undefined {
  if (diffRange.isEmpty()) {
    return undefined;
  }

  /**
   * diff range must be splited to single line before being mapped to editor range.
   */
  if (!diffRange.isSingleLine()) {
    return undefined;
  }

  const start: Position = {
    line: editorRanges[diffRange.startLineNumber - 1]?.start.line ?? 0,
    character: diffRange.startColumn - 1,
  };

  const end = {
    line: editorRanges[diffRange.startLineNumber - 1]?.start.line ?? 0,
    character: diffRange.endColumn - 1,
  };

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
      splitRangeToSingleLine(innerChange.originalRange, originCode).forEach((singleLineRange) => {
        const originRange = mapDiffRangeToEditorRange(singleLineRange, originCodeRanges);
        if (originRange) {
          originRanges.push(originRange);
        }
      });

      splitRangeToSingleLine(innerChange.modifiedRange, modifiedCode).forEach((singleLineRange) => {
        const modifiedRange = mapDiffRangeToEditorRange(singleLineRange, modifiedCodeRanges);
        if (modifiedRange) {
          modifiedRanges.push(modifiedRange);
        }
      });
    });
  });

  return {
    modifiedRanges,
    originRanges,
  };
}
