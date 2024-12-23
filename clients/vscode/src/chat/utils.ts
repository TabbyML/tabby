import path from "path";
import {
  Position as VSCodePosition,
  Range as VSCodeRange,
  Uri,
  workspace,
  DocumentSymbol,
  SymbolInformation,
  SymbolKind,
} from "vscode";
import type {
  Filepath,
  Position as ChatPanelPosition,
  LineRange,
  PositionRange,
  Location,
  SymbolAtInfo,
  FileAtInfo,
} from "tabby-chat-panel";
import type { GitProvider } from "../git/GitProvider";
import { getLogger } from "../logger";

const logger = getLogger("chat/utils");

export function localUriToChatPanelFilepath(uri: Uri, gitProvider: GitProvider): Filepath {
  const workspaceFolder = workspace.getWorkspaceFolder(uri);

  let repo = gitProvider.getRepository(uri);
  if (!repo && workspaceFolder) {
    repo = gitProvider.getRepository(workspaceFolder.uri);
  }
  const gitRemoteUrl = repo ? gitProvider.getDefaultRemoteUrl(repo) : undefined;

  if (repo && gitRemoteUrl) {
    const relativeFilePath = path.relative(repo.rootUri.toString(true), uri.toString(true));
    if (!relativeFilePath.startsWith("..")) {
      return {
        kind: "git",
        filepath: relativeFilePath,
        gitUrl: gitRemoteUrl,
      };
    }
  }

  return {
    kind: "uri",
    uri: uri.toString(true),
  };
}

export function chatPanelFilepathToLocalUri(filepath: Filepath, gitProvider: GitProvider): Uri | null {
  if (filepath.kind === "uri") {
    try {
      return Uri.parse(filepath.uri, true);
    } catch (e) {
      // FIXME(@icycodes): this is a hack for uri is relative filepaths in workspaces
      const workspaceRoot = workspace.workspaceFolders?.[0];
      if (workspaceRoot) {
        return Uri.joinPath(workspaceRoot.uri, filepath.uri);
      }
    }
  } else if (filepath.kind === "git") {
    const localGitRoot = gitProvider.findLocalRootUriByRemoteUrl(filepath.gitUrl);
    if (localGitRoot) {
      return Uri.joinPath(localGitRoot, filepath.filepath);
    }
  }
  logger.warn(`Invalid filepath params.`, filepath);
  return null;
}

export function vscodePositionToChatPanelPosition(position: VSCodePosition): ChatPanelPosition {
  return {
    line: position.line + 1,
    character: position.character + 1,
  };
}

export function chatPanelPositionToVSCodePosition(position: ChatPanelPosition): VSCodePosition {
  return new VSCodePosition(Math.max(0, position.line - 1), Math.max(0, position.character - 1));
}

export function vscodeRangeToChatPanelPositionRange(range: VSCodeRange): PositionRange {
  return {
    start: vscodePositionToChatPanelPosition(range.start),
    end: vscodePositionToChatPanelPosition(range.end),
  };
}

export function chatPanelPositionRangeToVSCodeRange(positionRange: PositionRange): VSCodeRange {
  return new VSCodeRange(
    chatPanelPositionToVSCodePosition(positionRange.start),
    chatPanelPositionToVSCodePosition(positionRange.end),
  );
}

export function chatPanelLineRangeToVSCodeRange(lineRange: LineRange): VSCodeRange {
  // Do not minus 1 from end line number, as we want to include the last line.
  return new VSCodeRange(Math.max(0, lineRange.start - 1), 0, lineRange.end, 0);
}

export function chatPanelLocationToVSCodeRange(location: Location): VSCodeRange | null {
  if (typeof location === "number") {
    const position = new VSCodePosition(Math.max(0, location - 1), 0);
    return new VSCodeRange(position, position);
  } else if ("line" in location) {
    const position = chatPanelPositionToVSCodePosition(location);
    return new VSCodeRange(position, position);
  } else if ("start" in location) {
    if (typeof location.start === "number") {
      return chatPanelLineRangeToVSCodeRange(location as LineRange);
    } else {
      return chatPanelPositionRangeToVSCodeRange(location as PositionRange);
    }
  }
  logger.warn(`Invalid location params.`, location);
  return null;
}

export function isDocumentSymbol(symbol: DocumentSymbol | SymbolInformation): symbol is DocumentSymbol {
  return "children" in symbol;
}

// FIXME: All allow symbol kinds, could be change later
export function getAllowedSymbolKinds(): SymbolKind[] {
  return [
    SymbolKind.Class,
    SymbolKind.Function,
    SymbolKind.Method,
    SymbolKind.Interface,
    SymbolKind.Enum,
    SymbolKind.Struct,
  ];
}

export function vscodeSymbolToAtInfo(
  symbol: DocumentSymbol | SymbolInformation,
  documentUri: Uri,
  gitProvider: GitProvider,
): SymbolAtInfo {
  if (isDocumentSymbol(symbol)) {
    return {
      atKind: "symbol",
      name: symbol.name,
      location: {
        filepath: localUriToChatPanelFilepath(documentUri, gitProvider),
        location: vscodeRangeToChatPanelPositionRange(symbol.range),
      },
    };
  }
  return {
    atKind: "symbol",
    name: symbol.name,
    location: {
      filepath: localUriToChatPanelFilepath(documentUri, gitProvider),
      location: vscodeRangeToChatPanelPositionRange(symbol.location.range),
    },
  };
}

export function uriToFileAtInfo(uri: Uri, gitProvider: GitProvider): FileAtInfo {
  return {
    atKind: "file",
    name: path.basename(uri.fsPath),
    filepath: localUriToChatPanelFilepath(uri, gitProvider),
  };
}
