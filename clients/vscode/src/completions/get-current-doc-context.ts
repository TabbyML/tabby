import * as vscode from "vscode";

import { detectMultiline } from "./detect-multiline";
import { getFirstLine, getLastLine, getNextNonEmptyLine, getPrevNonEmptyLine, lines } from "./text-processing";

export interface DocumentContext extends DocumentDependentContext, LinesContext {
  position: vscode.Position;
  multilineTrigger: string | null;
  multilineTriggerPosition: vscode.Position | null;
}

export interface DocumentDependentContext {
  prefix: string;
  suffix: string;
  /**
   * This is set when the document context is looking at the selected item in the
   * suggestion widget and injects the item into the prefix.
   */
  injectedPrefix: string | null;
  /**
   * @deprecated
   * will be removed after migrating `completionPostProcessLogger` to OpenTelemtry exporter.
   */
  completionPostProcessId?: string;
}

interface GetCurrentDocContextParams {
  document: vscode.TextDocument;
  position: vscode.Position;
  /* A number representing the maximum length of the prefix to get from the document. */
  maxPrefixLength: number;
  /* A number representing the maximum length of the suffix to get from the document. */
  maxSuffixLength: number;
  context?: vscode.InlineCompletionContext;
  dynamicMultilineCompletions: boolean;
}

/**
 * Get the current document context based on the cursor position in the current document.
 */
export function getCurrentDocContext(params: GetCurrentDocContextParams): DocumentContext {
  const { document, position, maxPrefixLength, maxSuffixLength, context, dynamicMultilineCompletions } = params;
  const offset = document.offsetAt(position);

  // TODO(philipp-spiess): This requires us to read the whole document. Can we limit our ranges
  // instead?
  const completePrefix = document.getText(new vscode.Range(new vscode.Position(0, 0), position));
  const completeSuffix = document.getText(new vscode.Range(position, document.positionAt(document.getText().length)));

  // Patch the document to contain the selected completion from the popup dialog already
  let completePrefixWithContextCompletion = completePrefix;
  let injectedPrefix = null;
  if (context?.selectedCompletionInfo) {
    const { range, text } = context.selectedCompletionInfo;
    // A selected completion info attempts to replace the specified range with the inserted text
    //
    // We assume that the end of the range equals the current position, otherwise this would not
    // inject a prefix
    if (range.end.character === position.character && range.end.line === position.line) {
      const lastLine = lines(completePrefix).at(-1)!;
      const beforeLastLine = completePrefix.slice(0, -lastLine.length);
      completePrefixWithContextCompletion = beforeLastLine + lastLine.slice(0, range.start.character) + text;
      injectedPrefix = completePrefixWithContextCompletion.slice(completePrefix.length);
      if (injectedPrefix === "") {
        injectedPrefix = null;
      }
    } else {
      console.warn("The selected completion info does not match the current position");
    }
  }

  const prefixLines = lines(completePrefixWithContextCompletion);
  const suffixLines = lines(completeSuffix);

  let prefix: string;
  if (offset > maxPrefixLength) {
    let total = 0;
    let startLine = prefixLines.length;
    for (let i = prefixLines.length - 1; i >= 0; i--) {
      if (total + prefixLines[i].length > maxPrefixLength) {
        break;
      }
      startLine = i;
      total += prefixLines[i].length;
    }
    prefix = prefixLines.slice(startLine).join("\n");
  } else {
    prefix = prefixLines.join("\n");
  }

  let totalSuffix = 0;
  let endLine = 0;
  for (let i = 0; i < suffixLines.length; i++) {
    if (totalSuffix + suffixLines[i].length > maxSuffixLength) {
      break;
    }
    endLine = i + 1;
    totalSuffix += suffixLines[i].length;
  }
  const suffix = suffixLines.slice(0, endLine).join("\n");

  return getDerivedDocContext({
    position,
    languageId: document.languageId,
    dynamicMultilineCompletions,
    documentDependentContext: {
      prefix,
      suffix,
      injectedPrefix,
    },
  });
}

interface GetDerivedDocContextParams {
  languageId: string;
  position: vscode.Position;
  documentDependentContext: DocumentDependentContext;
  dynamicMultilineCompletions: boolean;
}

/**
 * Calculates `DocumentContext` based on the existing prefix and suffix.
 * Used if the document context needs to be calculated for the updated text but there's no `document` instance for that.
 */
export function getDerivedDocContext(params: GetDerivedDocContextParams): DocumentContext {
  const { position, documentDependentContext, languageId, dynamicMultilineCompletions } = params;
  const linesContext = getLinesContext(documentDependentContext);

  return {
    ...documentDependentContext,
    ...linesContext,
    position,
    ...detectMultiline({
      docContext: { ...linesContext, ...documentDependentContext },
      languageId,
      dynamicMultilineCompletions,
      position,
    }),
  };
}

/**
 * Inserts a completion into a specific document context and computes the updated cursor position.
 *
 * This will insert the completion at the `position` outlined in the document context and will
 * replace the whole rest of the line with the completion. This means that if you have content in
 * the sameLineSuffix, it will be an empty string afterwards.
 *
 *
 * NOTE: This will always move the position to the _end_ of the line that the text was inserted at,
 *       regardless of whether the text was inserted before the sameLineSuffix.
 *
 *       When inserting `2` into: `f(1, █);`, the document context will look like this `f(1, 2);█`
 */
export function insertIntoDocContext(
  docContext: DocumentContext,
  insertText: string,
  languageId: string,
  dynamicMultilineCompletions: boolean = false,
): DocumentContext {
  const { position, prefix, currentLinePrefix, currentLineSuffix, suffix } = docContext;

  const insertedLines = lines(insertText);

  const updatedPosition =
    insertedLines.length <= 1
      ? new vscode.Position(position.line, currentLinePrefix.length + insertedLines[0].length)
      : new vscode.Position(position.line + insertedLines.length - 1, insertedLines.at(-1)!.length);

  return getDerivedDocContext({
    languageId,
    position: updatedPosition,
    dynamicMultilineCompletions,
    documentDependentContext: {
      prefix: prefix + insertText,
      suffix: suffix.slice(currentLineSuffix.length),
      injectedPrefix: null,
    },
  });
}

export interface LinesContext {
  /** Text before the cursor on the same line. */
  currentLinePrefix: string;
  /** Text after the cursor on the same line. */
  currentLineSuffix: string;

  prevNonEmptyLine: string;
  nextNonEmptyLine: string;
}

interface GetLinesContextParams {
  prefix: string;
  suffix: string;
}

function getLinesContext(params: GetLinesContextParams): LinesContext {
  const { prefix, suffix } = params;

  const currentLinePrefix = getLastLine(prefix);
  const currentLineSuffix = getFirstLine(suffix);

  const prevNonEmptyLine = getPrevNonEmptyLine(prefix);
  const nextNonEmptyLine = getNextNonEmptyLine(suffix);

  return {
    currentLinePrefix,
    currentLineSuffix,
    prevNonEmptyLine,
    nextNonEmptyLine,
  };
}
