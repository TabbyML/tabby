import TreeSitterParser from "web-tree-sitter";
import { LRUCache } from "lru-cache";
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

var treeSitterInitialized = false;

async function createParser(language: string): Promise<TreeSitterParser> {
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
  const langWasmPaths = isTest
    ? [process.cwd(), "wasm", `tree-sitter-${language}.wasm`]
    : [__dirname, "wasm", `tree-sitter-${language}.wasm`];
  parser.setLanguage(await TreeSitterParser.Language.load(require("path").join(...langWasmPaths)));
  return parser;
}

const parsers = new Map<string, TreeSitterParser>();

async function getParser(language: string): Promise<TreeSitterParser> {
  let parser = parsers.get(language);
  if (!parser) {
    parser = await createParser(language);
    parsers.set(language, parser);
  }
  return parser;
}

const treeCache = new LRUCache<string, TreeSitterParser.Tree>({
  max: 100,
});

export async function parse(text: string, filepath: string, language: string): Promise<TreeSitterParser.Tree> {
  const parser = await getParser(language);
  const cachedTree = treeCache.get(filepath);
  const tree = parser.parse(text, cachedTree);
  treeCache.set(filepath, tree);
  return tree;
}
