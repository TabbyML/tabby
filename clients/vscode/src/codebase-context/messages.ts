import { URI } from "vscode-uri";

import { ActiveTextEditorSelectionRange } from "../editor";

// tracked for telemetry purposes. Which context source provided this context file.
// embeddings: context file returned by the embeddings client
// user: context file provided by the user explicitly via chat input
// keyword: the context file returned from local keyword search
// editor: context file retrieved from the current editor
export type ContextFileSource = "embeddings" | "user" | "keyword" | "editor" | "filename" | "unified";

export type ContextFileType = "file" | "symbol";

export type SymbolKind = "class" | "function" | "method";

export interface ContextFile {
  // Name of the context
  // for file, this is usually the relative path
  // for symbol, this is usually the fuzzy name of the symbol
  fileName: string;

  content?: string;

  repoName?: string;
  revision?: string;

  // Location
  uri?: URI;
  path?: {
    basename?: string;
    dirname?: string;
    relative?: string;
  };
  range?: ActiveTextEditorSelectionRange;

  // metadata
  source?: ContextFileSource;
  type?: ContextFileType;
  kind?: SymbolKind;
}

export interface PreciseContext {
  symbol: {
    fuzzyName?: string;
  };
  hoverText: string[];
  definitionSnippet: string;
  filePath: string;
  range?: {
    startLine: number;
    startCharacter: number;
    endLine: number;
    endCharacter: number;
  };
}

export interface HoverContext {
  symbolName: string;
  sourceSymbolName?: string;
  content: string[];
  uri: string;
  range?: {
    startLine: number;
    startCharacter: number;
    endLine: number;
    endCharacter: number;
  };
}
