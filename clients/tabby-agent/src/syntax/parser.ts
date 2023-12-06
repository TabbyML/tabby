import path from "path";
import TreeSitterParser from "web-tree-sitter";
import { isTest } from "../env";

// https://code.visualstudio.com/docs/languages/identifiers
export const languagesConfigs: Record<string, string> = {
  javascript: "tsx",
  typescript: "tsx",
  javascriptreact: "tsx",
  typescriptreact: "tsx",
  python: "python",
  go: "go",
  rust: "rust",
  ruby: "ruby",
};

let treeSitterInitialized = false;

async function createParser(languageConfig: string): Promise<TreeSitterParser> {
  if (!treeSitterInitialized) {
    await TreeSitterParser.init({
      locateFile(scriptName: string, scriptDirectory: string) {
        const paths = isTest ? [scriptDirectory, scriptName] : [scriptDirectory, "wasm", scriptName];
        return path.join(...paths);
      },
    });
    treeSitterInitialized = true;
  }
  const parser = new TreeSitterParser();
  const langWasmPaths = isTest
    ? [process.cwd(), "wasm", `tree-sitter-${languageConfig}.wasm`]
    : [__dirname, "wasm", `tree-sitter-${languageConfig}.wasm`];
  parser.setLanguage(await TreeSitterParser.Language.load(path.join(...langWasmPaths)));
  return parser;
}

const parsers = new Map<string, TreeSitterParser>();

export async function getParser(languageConfig: string): Promise<TreeSitterParser> {
  let parser = parsers.get(languageConfig);
  if (!parser) {
    parser = await createParser(languageConfig);
    parsers.set(languageConfig, parser);
  }
  return parser;
}
