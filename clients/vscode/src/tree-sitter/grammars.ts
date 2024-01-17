/**
 * List of all supported languages that we have grammars and
 * lexems for. Note that enum values are copied from VSCode API,
 * if we want to make it work with different editors we should
 * enhance language detection.
 *
 * TODO: Decouple language detect to make it editor agnostic
 */
export enum SupportedLanguage {
  JavaScript = "javascript",
  JSX = "javascriptreact",
  TypeScript = "typescript",
  TSX = "typescriptreact",
  Go = "go",
  Python = "python",
}

export const getParseLanguage = (languageId: string): SupportedLanguage | null => {
  const matchedLang = Object.entries(SupportedLanguage).find(
    ([, value]) => value === (languageId as SupportedLanguage),
  );

  return matchedLang ? (languageId as SupportedLanguage) : null;
};
