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

    // Get document symbols from the specified file
    const rawDocumentSymbols =
      (await commands.executeCommand<DocumentSymbol[] | SymbolInformation[]>(
        "vscode.executeDocumentSymbolProvider",
        uri,
      )) || [];

    // Convert DocumentSymbol[] to SymbolInformation[] if needed
    const documentSymbols = convertToSymbolInformation(rawDocumentSymbols);

    // Filter symbols from the current file only
    const filteredSymbols = filterSymbols(documentSymbols, queryString, maxResults);

    // Convert to result format with appropriate icons
    return filteredSymbols.map((symbol) => ({
      name: symbol.name,
      kind: symbol.kind,
      location: symbol.location,
      kindIcon: getSymbolIcon(symbol.kind),
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

  // Check if we have DocumentSymbol[] or SymbolInformation[]
  if (symbols.length > 0 && symbols[0] && "children" in symbols[0] && window.activeTextEditor) {
    // We have DocumentSymbol[], need to flatten
    flattenDocumentSymbols(symbols as DocumentSymbol[], "", window.activeTextEditor.document.uri, result);
  } else if (symbols.length > 0) {
    // We already have SymbolInformation[]
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

    // Create a SymbolInformation from DocumentSymbol
    result.push({
      name: symbol.name,
      kind: symbol.kind,
      containerName: containerName,
      location: {
        uri: uri,
        range: symbol.range,
      },
    } as SymbolInformation);

    // Process children recursively
    if (symbol.children && symbol.children.length > 0) {
      flattenDocumentSymbols(symbol.children, fullName, uri, result);
    }
  }
}

/**
 * Maps symbol kinds to appropriate theme icons
 */
function getSymbolIcon(kind: SymbolKind): ThemeIcon {
  // Map symbol kinds to appropriate theme icons
  switch (kind) {
    case SymbolKind.File:
      return new ThemeIcon("file");
    case SymbolKind.Module:
      return new ThemeIcon("package");
    case SymbolKind.Namespace:
      return new ThemeIcon("symbol-namespace");
    case SymbolKind.Class:
      return new ThemeIcon("symbol-class");
    case SymbolKind.Method:
      return new ThemeIcon("symbol-method");
    case SymbolKind.Property:
      return new ThemeIcon("symbol-property");
    case SymbolKind.Field:
      return new ThemeIcon("symbol-field");
    case SymbolKind.Constructor:
      return new ThemeIcon("symbol-constructor");
    case SymbolKind.Enum:
      return new ThemeIcon("symbol-enum");
    case SymbolKind.Interface:
      return new ThemeIcon("symbol-interface");
    case SymbolKind.Function:
      return new ThemeIcon("symbol-method");
    case SymbolKind.Variable:
      return new ThemeIcon("symbol-variable");
    case SymbolKind.Constant:
      return new ThemeIcon("symbol-constant");
    case SymbolKind.String:
      return new ThemeIcon("symbol-string");
    case SymbolKind.Number:
      return new ThemeIcon("symbol-number");
    case SymbolKind.Boolean:
      return new ThemeIcon("symbol-boolean");
    case SymbolKind.Array:
      return new ThemeIcon("symbol-array");
    case SymbolKind.Object:
      return new ThemeIcon("symbol-object");
    case SymbolKind.Key:
      return new ThemeIcon("symbol-key");
    case SymbolKind.Null:
      return new ThemeIcon("symbol-null");
    case SymbolKind.EnumMember:
      return new ThemeIcon("symbol-enum-member");
    case SymbolKind.Struct:
      return new ThemeIcon("symbol-struct");
    case SymbolKind.Event:
      return new ThemeIcon("symbol-event");
    case SymbolKind.Operator:
      return new ThemeIcon("symbol-operator");
    case SymbolKind.TypeParameter:
      return new ThemeIcon("symbol-parameter");
    default:
      return new ThemeIcon("symbol-misc");
  }
}

/**
 * Filters and sorts symbols based on the query
 */
function filterSymbols(symbols: SymbolInformation[], query: string, maxResults: number): SymbolInformation[] {
  // Remove duplicates
  const uniqueSymbols = removeDuplicateSymbols(symbols);

  if (!query || !window.activeTextEditor) {
    return uniqueSymbols.slice(0, maxResults);
  }
  const editor = window.activeTextEditor;

  const lowerQuery = query.toLowerCase();
  const filtered = uniqueSymbols.filter((s) => s.name.toLowerCase().includes(lowerQuery));

  // Sort by match quality
  return filtered
    .sort((a, b) => {
      const aName = a.name.toLowerCase();
      const bName = b.name.toLowerCase();

      // Exact match
      if (aName === lowerQuery && bName !== lowerQuery) return -1;
      if (bName === lowerQuery && aName !== lowerQuery) return 1;

      // Starts with query
      if (aName.startsWith(lowerQuery) && !bName.startsWith(lowerQuery)) return -1;
      if (bName.startsWith(lowerQuery) && !aName.startsWith(lowerQuery)) return 1;

      // Current file symbols first
      const aIsCurrentFile = a.location.uri.toString() === editor.document.uri.toString();
      const bIsCurrentFile = b.location.uri.toString() === editor.document.uri.toString();
      if (aIsCurrentFile && !bIsCurrentFile) return -1;
      if (bIsCurrentFile && !aIsCurrentFile) return 1;

      // Shorter name is better
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
