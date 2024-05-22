import { commands, Position, Range, SemanticTokens, SemanticTokensLegend, TextDocument, Webview, Uri } from "vscode";

export type SemanticSymbolInfo = {
  position: Position;
  type: string;
};

// reference: https://code.visualstudio.com/api/language-extensions/semantic-highlight-guide
export async function extractSemanticSymbols(
  document: TextDocument,
  range: Range,
): Promise<SemanticSymbolInfo[] | undefined> {
  const providedTokens = await commands.executeCommand(
    "vscode.provideDocumentRangeSemanticTokens",
    document.uri,
    range,
  );
  if (
    typeof providedTokens === "object" &&
    providedTokens !== null &&
    "resultId" in providedTokens &&
    "data" in providedTokens
  ) {
    const tokens = providedTokens as SemanticTokens;
    const providedLegend = await commands.executeCommand(
      "vscode.provideDocumentRangeSemanticTokensLegend",
      document.uri,
      range,
    );
    if (
      typeof providedLegend === "object" &&
      providedLegend !== null &&
      "tokenTypes" in providedLegend &&
      "tokenModifiers" in providedLegend
    ) {
      const legend = providedLegend as SemanticTokensLegend;

      const semanticSymbols: SemanticSymbolInfo[] = [];
      let line = 0;
      let char = 0;
      for (let i = 0; i + 4 < tokens.data.length; i += 5) {
        const deltaLine = tokens.data[i]!;
        const deltaChar = tokens.data[i + 1]!;
        // i + 2 is token length, not used here
        const type = legend.tokenTypes[tokens.data[i + 3]!] ?? "";
        // i + 4 is type modifiers, not used here

        line += deltaLine;
        if (deltaLine > 0) {
          char = deltaChar;
        } else {
          char += deltaChar;
        }
        semanticSymbols.push({
          position: new Position(line, char),
          type,
        });
      }
      return semanticSymbols;
    }
  }
  return undefined;
}

// Keywords appear in the code everywhere, but we don't want to use them for
// matching in code searching.
// Just filter them out before we start using a syntax parser.
const reservedKeywords = [
  // Typescript: https://github.com/microsoft/TypeScript/issues/2536
  "as",
  "any",
  "boolean",
  "break",
  "case",
  "catch",
  "class",
  "const",
  "constructor",
  "continue",
  "debugger",
  "declare",
  "default",
  "delete",
  "do",
  "else",
  "enum",
  "export",
  "extends",
  "false",
  "finally",
  "for",
  "from",
  "function",
  "get",
  "if",
  "implements",
  "import",
  "in",
  "instanceof",
  "interface",
  "let",
  "module",
  "new",
  "null",
  "number",
  "of",
  "package",
  "private",
  "protected",
  "public",
  "require",
  "return",
  "set",
  "static",
  "string",
  "super",
  "switch",
  "symbol",
  "this",
  "throw",
  "true",
  "try",
  "typeof",
  "var",
  "void",
  "while",
  "with",
  "yield",
];
export function extractNonReservedWordList(text: string): string {
  const re = /\w+/g;
  return [
    ...new Set(text.match(re)?.filter((symbol) => symbol.length > 2 && !reservedKeywords.includes(symbol))).values(),
  ].join(" ");
}

/**
 * A helper function which will get the webview URI of a given file or resource.
 */
export function getUri(webview: Webview, extensionUri: Uri, pathList: string[]) {
  return webview.asWebviewUri(Uri.joinPath(extensionUri, ...pathList));
}
