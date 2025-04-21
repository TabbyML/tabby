import type { Position, Range, SelectedCompletionInfo } from "vscode-languageserver";
import { TextDocument } from "vscode-languageserver-textdocument";
import { splitLines } from "../utils/string";
import { documentRange, rangeInDocument } from "../utils/range";
import { getLogger } from "../logger";
import { CodeSearchResult } from "../codeSearch";
import { TextDocumentRangeContext } from "../contextProviders/documentContexts";
import { WorkspaceContext } from "../contextProviders/workspace";
import { GitContext } from "../contextProviders/git";
import { EditorOptionsContext } from "../contextProviders/editorOptions";

const logger = getLogger("CodeCompletionContext");

export interface CompletionContext {
  readonly document: TextDocument;
  readonly position: Position;
  readonly selectedCompletionInfo?: SelectedCompletionInfo;
  readonly notebookCells?: TextDocument[];

  // calculated from selectedCompletionInfo, this insertion text is already included in prefix
  readonly selectedCompletionInsertion: string;

  // the line suffix is empty or should be replaced, in this case, the line suffix is already excluded from suffix
  readonly isLineEnd: boolean;
  readonly lineEndReplaceLength: number;

  // calculated from contexts, do not equal to document prefix and suffix
  readonly prefix: string;
  readonly suffix: string;

  // redundant quick access for prefix and suffix
  readonly prefixLines: string[];
  readonly suffixLines: string[];
  readonly currentLinePrefix: string;
  readonly currentLineSuffix: string;
}

export function buildCompletionContext(
  document: TextDocument,
  position: Position,
  selectedCompletionInfo?: SelectedCompletionInfo,
  notebookCells?: TextDocument[],
): CompletionContext {
  let selectedCompletionInsertion = "";
  if (selectedCompletionInfo) {
    // Handle selected completion info only if replacement matches prefix
    // Handle: con -> console
    // Ignore: cns -> console
    const replaceRange = converToObjectRange(selectedCompletionInfo.range);
    if (
      replaceRange.start.line == position.line &&
      replaceRange.start.character < position.character &&
      replaceRange.end.line == position.line &&
      replaceRange.end.character == position.character
    ) {
      const replaceLength = replaceRange.end.character - replaceRange.start.character;
      selectedCompletionInsertion = selectedCompletionInfo.text.substring(replaceLength);
      logger.trace("Used selected completion insertion: ", { selectedCompletionInsertion });
    }
  }

  let notebookCellsPrefix = "";
  let notebookCellsSuffix = "";
  if (notebookCells) {
    const currentCellIndex = notebookCells.indexOf(document);
    if (currentCellIndex >= 0 && currentCellIndex < notebookCells.length - 1) {
      const currentLanguageId = document.languageId;
      const formatContext = (cells: TextDocument[]): string => {
        const notebookLanguageComments: { [languageId: string]: (code: string) => string } = {
          markdown: (code) => "```\n" + code + "\n```",
          python: (code) =>
            code
              .split("\n")
              .map((l) => "# " + l)
              .join("\n"),
        };
        return cells
          .map((textDocument) => {
            if (textDocument.languageId === currentLanguageId) {
              return textDocument.getText();
            } else if (Object.keys(notebookLanguageComments).includes(currentLanguageId)) {
              return notebookLanguageComments[currentLanguageId]?.(textDocument.getText()) ?? "";
            } else {
              return "";
            }
          })
          .join("\n\n");
      };
      notebookCellsPrefix = formatContext(notebookCells.slice(0, currentCellIndex)) + "\n\n";
      notebookCellsSuffix = "\n\n" + formatContext(notebookCells.slice(currentCellIndex + 1));
      logger.trace("Used notebook cells context:", { notebookCellsPrefix, notebookCellsSuffix });
    }
  }

  const fullDocumentRange = documentRange(document);
  const prefixRange = {
    start: fullDocumentRange.start,
    end: position,
  };
  const documentPrefix = document.getText(prefixRange);
  const prefix = notebookCellsPrefix + documentPrefix + selectedCompletionInsertion;

  const documentCurrentLineSuffixRange = rangeInDocument(
    {
      start: position,
      end: { line: position.line + 1, character: 0 },
    },
    document,
  );
  const documentCurrentLineSuffix = documentCurrentLineSuffixRange
    ? document.getText(documentCurrentLineSuffixRange)
    : "";
  const isLineEnd = !!documentCurrentLineSuffix.match(/^\W*$/);
  const lineEndReplaceLength = isLineEnd ? documentCurrentLineSuffix.replace(/\r?\n$/, "").length : 0;

  const suffixRange = rangeInDocument(
    {
      start: { line: position.line, character: position.character + lineEndReplaceLength },
      end: fullDocumentRange.end,
    },
    document,
  );
  const documentSuffix = suffixRange ? document.getText(suffixRange) : "";

  const suffix = documentSuffix + notebookCellsSuffix;

  const prefixLines = splitLines(prefix);
  const suffixLines = splitLines(suffix);
  const currentLinePrefix = prefixLines[prefixLines.length - 1] ?? "";
  const currentLineSuffix = suffixLines[0] ?? "";

  return {
    document,
    position,
    selectedCompletionInfo,
    notebookCells,
    selectedCompletionInsertion,
    isLineEnd,
    lineEndReplaceLength,
    prefix,
    suffix,
    prefixLines,
    suffixLines,
    currentLinePrefix,
    currentLineSuffix,
  };
}

export function buildCompletionContextWithAppend(context: CompletionContext, appendText: string): CompletionContext {
  const offset = context.document.offsetAt(context.position);
  const updatedText = context.prefix + appendText + context.suffix;
  const updatedOffset = offset + appendText.length;
  const updatedDocument = TextDocument.create(
    context.document.uri,
    context.document.languageId,
    context.document.version + 1,
    updatedText,
  );
  const updatedPosition = updatedDocument.positionAt(updatedOffset);
  return buildCompletionContext(updatedDocument, updatedPosition, undefined, context.notebookCells);
}

export interface CompletionExtraContexts {
  workspace?: WorkspaceContext;
  git?: GitContext;
  declarations?: TextDocumentRangeContext[];
  recentlyChangedCodeSearchResult?: CodeSearchResult;
  lastViewedSnippets?: TextDocumentRangeContext[];
  editorOptions?: EditorOptionsContext;
}

function converToObjectRange(range: Range | [Position, Position]): Range {
  if (Array.isArray(range)) {
    return {
      start: range[0],
      end: range[1],
    };
  }
  return range;
}
