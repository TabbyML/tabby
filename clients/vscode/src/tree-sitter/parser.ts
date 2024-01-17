import path, { join } from "path";

import Parser from "web-tree-sitter";

import { SupportedLanguage } from "./grammars";
import { initQueries } from "./query-sdk";
import TreeSitterParser from "web-tree-sitter";

/*
 * Loading wasm grammar and creation parser instance everytime we trigger
 * pre- and post-process might be a performance problem, so we create instance
 * and load language grammar only once, first time we need parser for a specific
 * language, next time we read it from this cache.
 */
const PARSERS_LOCAL_CACHE: Partial<Record<SupportedLanguage, Parser>> = {};

interface ParserSettings {
  language: SupportedLanguage;

  /*
   * A custom path to the directory where we store wasm grammar modules
   * primary reasons for this is to provide a custom path for testing
   */
  grammarDirectory?: string;
}

export function getParser(language: SupportedLanguage): Parser | undefined {
  return PARSERS_LOCAL_CACHE[language];
}

export function resetParsersCache(): void {
  for (const key of Object.keys(PARSERS_LOCAL_CACHE)) {
    delete PARSERS_LOCAL_CACHE[key as SupportedLanguage];
  }
}

export async function createParser(settings: ParserSettings): Promise<Parser> {
  const { language } = settings;
  const grammarDirectory = settings.grammarDirectory ?? join(__dirname, "wasm");

  const cachedParser = PARSERS_LOCAL_CACHE[language];

  if (cachedParser) {
    return cachedParser;
  }

  try {
    await TreeSitterParser.init({
      locateFile(scriptName: string, scriptDirectory: string) {
        const paths = [scriptDirectory, "wasm", scriptName];
        return path.join(...paths);
      },
    });
    const parser = new TreeSitterParser();

    const wasmPath = path.resolve(grammarDirectory, SUPPORTED_LANGUAGES[language]);

    const languageGrammar = await TreeSitterParser.Language.load(wasmPath);
    parser.setLanguage(languageGrammar);
    PARSERS_LOCAL_CACHE[language] = parser;

    initQueries(languageGrammar, language, parser);

    return parser;
  } catch (e) {
    console.error("cant load wasm", e);
    throw e;
  }
}

// TODO: Add grammar type autogenerate script
// see https://github.com/asgerf/dts-tree-sitter
type GrammarPath = string;

/**
 * Map language to wasm grammar path modules, usually we would have
 * used node bindings for grammar packages, but since VSCode editor
 * runtime doesn't support this we have to work with wasm modules.
 *
 * Note: make sure that dist folder contains these modules when you
 * run VSCode extension.
 */
const SUPPORTED_LANGUAGES: Record<SupportedLanguage, GrammarPath> = {
  [SupportedLanguage.JavaScript]: "tree-sitter-tsx.wasm",
  [SupportedLanguage.JSX]: "tree-sitter-tsx.wasm",
  [SupportedLanguage.TypeScript]: "tree-sitter-tsx.wasm",
  [SupportedLanguage.TSX]: "tree-sitter-tsx.wasm",
  [SupportedLanguage.Go]: "tree-sitter-go.wasm",
  [SupportedLanguage.Python]: "tree-sitter-python.wasm",
};
