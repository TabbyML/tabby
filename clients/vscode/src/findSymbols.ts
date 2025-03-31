import { commands, DocumentSymbol, SymbolInformation, Uri, SymbolKind, ThemeIcon, window, Location } from "vscode";
import { getLogger } from "./logger";

const logger = getLogger("findSymbols");

/**
 * Interface definition for symbol information returned by listSymbols
 */
export interface SymbolInfo {
  name: string;
  kind: SymbolKind;
  location: Location;
  kindIcon: ThemeIcon;
  containerName?: string;
}

/**
 * Fetches symbols from the current file based on the query
 * @param uri The URI of the file to search in
 * @param query The search query
 * @param limit Maximum number of results to return
 * @returns Array of symbol information with additional metadata
 */
export async function listSymbols(uri: Uri, query: string, limit?: number): Promise<SymbolInfo[]> {
  try {
    const maxResults = limit ?? 50;
    const queryString = query.trim().toLowerCase();
    const rawDocumentSymbols =
      (await commands.executeCommand<DocumentSymbol[] | SymbolInformation[]>(
        "vscode.executeDocumentSymbolProvider",
        uri,
      )) || [];

    const documentSymbols = convertToSymbolInformation(rawDocumentSymbols);

    const filteredSymbols = filterSymbols(documentSymbols, queryString, maxResults);

    return filteredSymbols.map((symbol) => ({
      name: symbol.name,
      kind: symbol.kind,
      location: symbol.location,
      kindIcon: new ThemeIcon(symbolIconMap.get(symbol.kind) ?? "symbol-misc"),
      containerName: symbol.containerName,
    }));
  } catch (error) {
    logger.debug("Failed to find symbols:", error);
    return [];
  }
}

/**
 * Converts DocumentSymbol[] to SymbolInformation[] for consistent handling
 */
function convertToSymbolInformation(symbols: (DocumentSymbol | SymbolInformation)[]): SymbolInformation[] {
  const result: SymbolInformation[] = [];

  if (symbols.length > 0 && symbols[0] && "children" in symbols[0] && window.activeTextEditor) {
    flattenDocumentSymbols(symbols as DocumentSymbol[], "", window.activeTextEditor.document.uri, result);
  } else if (symbols.length > 0) {
    result.push(...(symbols as SymbolInformation[]));
  }

  return result;
}

/**
 * Flattens a hierarchical DocumentSymbol structure into a flat SymbolInformation array
 */
function flattenDocumentSymbols(
  symbols: DocumentSymbol[],
  containerName: string,
  uri: Uri,
  result: SymbolInformation[],
): void {
  for (const symbol of symbols) {
    const fullName = containerName ? `${containerName}.${symbol.name}` : symbol.name;

    result.push({
      name: symbol.name,
      kind: symbol.kind,
      containerName: containerName,
      location: {
        uri: uri,
        range: symbol.range,
      },
    } as SymbolInformation);

    if (symbol.children && symbol.children.length > 0) {
      flattenDocumentSymbols(symbol.children, fullName, uri, result);
    }
  }
}

/**
 * Maps symbol kinds to appropriate theme icons name
 */
const symbolIconMap = new Map<SymbolKind, string>([
  [SymbolKind.File, "file"],
  [SymbolKind.Module, "package"],
  [SymbolKind.Namespace, "symbol-namespace"],
  [SymbolKind.Class, "symbol-class"],
  [SymbolKind.Method, "symbol-method"],
  [SymbolKind.Property, "symbol-property"],
  [SymbolKind.Field, "symbol-field"],
  [SymbolKind.Constructor, "symbol-constructor"],
  [SymbolKind.Enum, "symbol-enum"],
  [SymbolKind.Interface, "symbol-interface"],
  [SymbolKind.Function, "symbol-method"],
  [SymbolKind.Variable, "symbol-variable"],
  [SymbolKind.Constant, "symbol-constant"],
  [SymbolKind.String, "symbol-string"],
  [SymbolKind.Number, "symbol-number"],
  [SymbolKind.Boolean, "symbol-boolean"],
  [SymbolKind.Array, "symbol-array"],
  [SymbolKind.Object, "symbol-object"],
  [SymbolKind.Key, "symbol-key"],
  [SymbolKind.Null, "symbol-null"],
  [SymbolKind.EnumMember, "symbol-enum-member"],
  [SymbolKind.Struct, "symbol-struct"],
  [SymbolKind.Event, "symbol-event"],
  [SymbolKind.Operator, "symbol-operator"],
  [SymbolKind.TypeParameter, "symbol-parameter"],
]);

/**
 * Filters and sorts symbols based on the query
 */
function filterSymbols(symbols: SymbolInformation[], query: string, maxResults: number): SymbolInformation[] {
  const uniqueSymbols = removeDuplicateSymbols(symbols);

  if (!query || !window.activeTextEditor) {
    return uniqueSymbols.slice(0, maxResults);
  }
  const editor = window.activeTextEditor;

  const lowerQuery = query.toLowerCase();
  const filtered = uniqueSymbols.filter((s) => s.name.toLowerCase().includes(lowerQuery));
  return filtered
    .sort((a, b) => {
      const aName = a.name.toLowerCase();
      const bName = b.name.toLowerCase();
      if (aName === lowerQuery && bName !== lowerQuery) return -1;
      if (bName === lowerQuery && aName !== lowerQuery) return 1;
      if (aName.startsWith(lowerQuery) && !bName.startsWith(lowerQuery)) return -1;
      if (bName.startsWith(lowerQuery) && !aName.startsWith(lowerQuery)) return 1;
      const aIsCurrentFile = a.location.uri.toString() === editor.document.uri.toString();
      const bIsCurrentFile = b.location.uri.toString() === editor.document.uri.toString();
      if (aIsCurrentFile && !bIsCurrentFile) return -1;
      if (bIsCurrentFile && !aIsCurrentFile) return 1;
      return a.name.length - b.name.length;
    })
    .slice(0, maxResults);
}

/**
 * Removes duplicate symbols from the list
 */
function removeDuplicateSymbols(symbols: SymbolInformation[]): SymbolInformation[] {
  const seen = new Set<string>();
  return symbols.filter((symbol) => {
    const key = `${symbol.name}-${symbol.containerName}-${symbol.location.uri.toString()}`;
    if (seen.has(key)) {
      return false;
    }
    seen.add(key);
    return true;
  });
}
