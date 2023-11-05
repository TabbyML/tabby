import TreeSitterParser from "web-tree-sitter";
import { typesList } from "./treeSitterTypesList";
import { LRUCache } from "lru-cache";
import { CompletionContext } from "../Agent";
import { PostprocessFilter, logger } from "./base";
import { isTest } from "../env";

// https://code.visualstudio.com/docs/languages/identifiers
const languagesConfig: Record<string, string> = {
  javascript: "tsx",
  typescript: "tsx",
  javascriptreact: "tsx",
  typescriptreact: "tsx",
  python: "python",
  go: "go",
  rust: "rust",
  ruby: "ruby",
};

export const supportedLanguages = Object.keys(languagesConfig);

const treeCache = new LRUCache<string, TreeSitterParser.Tree>({
  max: 10,
});

const parsers = new Map<string, TreeSitterParser>();

var treeSitterInitialized = false;

async function createParser(config: string): Promise<TreeSitterParser> {
  if (!treeSitterInitialized) {
    await TreeSitterParser.init({
      locateFile(scriptName: string, scriptDirectory: string) {
        const paths = isTest ? [scriptDirectory, scriptName] : [scriptDirectory, "wasm", scriptName];
        return require("path").join(...paths);
      },
    });
    treeSitterInitialized = true;
  }
  const parser = new TreeSitterParser();
  const langPaths = isTest
    ? [process.cwd(), "wasm", `tree-sitter-${config}.wasm`]
    : [__dirname, "wasm", `tree-sitter-${config}.wasm`];
  parser.setLanguage(await TreeSitterParser.Language.load(require("path").join(...langPaths)));
  return parser;
}

async function getParser(config: string): Promise<TreeSitterParser> {
  let parser = parsers.get(config);
  if (!parser) {
    parser = await createParser(config);
    parsers.set(config, parser);
  }
  return parser;
}

function findLineBegin(text: string, position: number): number {
  let lastNonBlankCharPos = position - 1;
  while (lastNonBlankCharPos >= 0 && text[lastNonBlankCharPos].match(/\s/)) {
    lastNonBlankCharPos--;
  }
  if (lastNonBlankCharPos < 0) {
    return 0;
  }
  const lineBegin = text.lastIndexOf("\n", lastNonBlankCharPos);
  if (lineBegin < 0) {
    return 0;
  }
  const line = text.slice(lineBegin + 1, position);
  const indentation = line.search(/\S/);
  return lineBegin + 1 + indentation;
}

function findLineEnd(text: string, position: number): number {
  let firstNonBlankCharPos = position;
  while (firstNonBlankCharPos < text.length && text[firstNonBlankCharPos].match(/\s/)) {
    firstNonBlankCharPos++;
  }
  if (firstNonBlankCharPos >= text.length) {
    return text.length;
  }
  const lineEnd = text.indexOf("\n", firstNonBlankCharPos);
  if (lineEnd < 0) {
    return text.length;
  }
  return lineEnd;
}

function findScope(node: TreeSitterParser.SyntaxNode, typesList: string[][]): TreeSitterParser.SyntaxNode {
  for (const types of typesList) {
    let scope = node;
    while (scope) {
      if (types.indexOf(scope.type) >= 0) {
        return scope;
      }
      scope = scope.parent;
    }
  }
  return node;
}

export function limitScopeBySyntax(context: CompletionContext): PostprocessFilter {
  return async (input) => {
    const { position, text, language, filepath, prefix, suffix } = context;
    if (supportedLanguages.indexOf(language) < 0) {
      return input;
    }
    const config = languagesConfig[language];
    const parser = await getParser(config);
    const types = typesList[config];

    const cachedTree = treeCache.get(filepath);
    const tree = parser.parse(text, cachedTree);
    treeCache.set(filepath, tree);

    const updatedText = prefix + input + suffix;
    const updatedTree = parser.parse(updatedText, cachedTree);
    const lineBegin = findLineBegin(updatedText, position);
    const lineEnd = findLineEnd(updatedText, position);
    const scope = findScope(updatedTree.rootNode.namedDescendantForIndex(lineBegin, lineEnd), types);

    if (scope.endIndex < position + input.length) {
      logger.debug(
        { text, updatedText, scope: { type: scope.type, start: scope.startIndex, end: scope.endIndex } },
        "Remove content out of syntax scope",
      );
      return input.slice(0, scope.endIndex - position);
    }
    return input;
  };
}
