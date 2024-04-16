export function getWordStartIndices(text: string): number[] {
  const indices: number[] = [];
  const re = /\b\w/g;
  let match;
  while ((match = re.exec(text)) != null) {
    indices.push(match.index);
  }
  return indices;
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
export function extractSematicSymbols(text: string): string {
  const re = /\w+/g;
  return [
    ...new Set(text.match(re)?.filter((symbol) => symbol.length > 2 && !reservedKeywords.includes(symbol))).values(),
  ].join(" ");
}
