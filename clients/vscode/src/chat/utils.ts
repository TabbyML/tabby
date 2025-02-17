import path from "path";
import { TextEditor, Position as VSCodePosition, Range as VSCodeRange, Uri, workspace } from "vscode";
import type {
  Filepath,
  Position as ChatPanelPosition,
  LineRange,
  PositionRange,
  Location,
  ListFileItem,
} from "tabby-chat-panel";
import type { GitProvider } from "../git/GitProvider";
import { getLogger } from "../logger";

const logger = getLogger("chat/utils");

enum DocumentSchemes {
  file = "file",
  untitled = "untitled",
  vscodeNotebookCell = "vscode-notebook-cell",
  vscodeVfs = "vscode-vfs",
}

export function isValidForSyncActiveEditorSelection(editor: TextEditor): boolean {
  const supportedSchemes: string[] = [
    DocumentSchemes.file,
    DocumentSchemes.untitled,
    DocumentSchemes.vscodeNotebookCell,
    DocumentSchemes.vscodeVfs,
  ];
  return supportedSchemes.includes(editor.document.uri.scheme);
}

export function localUriToChatPanelFilepath(uri: Uri, gitProvider: GitProvider): Filepath {
  let localUri = uri;
  if (localUri.scheme === DocumentSchemes.vscodeNotebookCell) {
    localUri = convertFromNotebookCellUri(localUri);
  }

  const uriFilePath = localUri.toString(true);
  const workspaceFolder = workspace.getWorkspaceFolder(localUri);

  let repo = gitProvider.getRepository(localUri);
  if (!repo && workspaceFolder) {
    repo = gitProvider.getRepository(workspaceFolder.uri);
  }
  const gitRemoteUrl = repo ? gitProvider.getDefaultRemoteUrl(repo) : undefined;
  if (repo && gitRemoteUrl) {
    const relativeFilePath = path.relative(repo.rootUri.toString(true), uriFilePath);
    if (!relativeFilePath.startsWith("..")) {
      return {
        kind: "git",
        filepath: relativeFilePath,
        gitUrl: gitRemoteUrl,
      };
    }
  }

  if (workspaceFolder) {
    const baseDir = workspaceFolder.uri.toString(true);
    const relativeFilePath = path.relative(baseDir, uriFilePath);
    if (!relativeFilePath.startsWith("..")) {
      return {
        kind: "workspace",
        filepath: relativeFilePath,
        baseDir: baseDir,
      };
    }
  }

  return {
    kind: "uri",
    uri: uriFilePath,
  };
}

export function chatPanelFilepathToLocalUri(filepath: Filepath, gitProvider: GitProvider): Uri | null {
  let result: Uri | null = null;
  if (filepath.kind === "uri") {
    try {
      result = Uri.parse(filepath.uri, true);
    } catch (e) {
      const workspaceRoot = workspace.workspaceFolders?.[0];
      if (workspaceRoot) {
        result = Uri.joinPath(workspaceRoot.uri, filepath.uri);
      }
    }
  } else if (filepath.kind === "workspace") {
    try {
      const workspaceFolder = workspace.getWorkspaceFolder(Uri.parse(filepath.baseDir, true));
      if (workspaceFolder) {
        result = Uri.joinPath(workspaceFolder.uri, filepath.filepath);
      }
    } catch (e) {
      // do nothing
    }
  } else if (filepath.kind === "git") {
    const localGitRoot = gitProvider.findLocalRootUriByRemoteUrl(filepath.gitUrl);
    if (localGitRoot) {
      result = Uri.joinPath(localGitRoot, filepath.filepath);
    }
  }

  if (result == null) {
    logger.warn(`Invalid filepath params.`, filepath);
    return null;
  }

  if (isJupyterNotebookFilepath(result)) {
    result = convertToNotebookCellUri(result);
  }
  return result;
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

export function vscodeRangeToChatPanelLineRange(range: VSCodeRange): LineRange {
  return {
    start: range.start.line + 1,
    end: range.end.line + 1,
  };
}

export function chatPanelLocationToVSCodeRange(location: Location | undefined): VSCodeRange | null {
  if (!location) {
    return null;
  }
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

export function localUriToListFileItem(uri: Uri, gitProvider: GitProvider): ListFileItem {
  return {
    filepath: localUriToChatPanelFilepath(uri, gitProvider),
  };
}

// Notebook cell uri conversion

function isJupyterNotebookFilepath(uri: Uri): boolean {
  const extname = path.extname(uri.fsPath);
  return extname.startsWith(".ipynb");
}

function convertToNotebookCellUri(uri: Uri): Uri {
  let handle: number | undefined;

  const searchParams = new URLSearchParams(uri.fragment);
  const cellString = searchParams.get("cell");
  if (cellString) {
    handle = parseInt(cellString, 10);
  }
  handle = handle || 0;

  searchParams.set("cell", handle.toString());
  return generateNotebookCellUri(uri, handle);
}

function convertFromNotebookCellUri(uri: Uri): Uri {
  const parsed = parseNotebookCellUri(uri);
  if (!parsed) {
    return uri;
  }
  return uri.with({ scheme: parsed.notebook.scheme, fragment: `cell=${parsed.handle}` });
}

const nb_lengths = ["W", "X", "Y", "Z", "a", "b", "c", "d", "e", "f"];
const nb_padRegexp = new RegExp(`^[${nb_lengths.join("")}]+`);
const nb_radix = 7;

function parseNotebookCellUri(cell: Uri): { notebook: Uri; handle: number } | undefined {
  if (cell.scheme !== DocumentSchemes.vscodeNotebookCell) {
    return undefined;
  }

  const idx = cell.fragment.indexOf("s");
  if (idx < 0) {
    return undefined;
  }

  const handle = parseInt(cell.fragment.substring(0, idx).replace(nb_padRegexp, ""), nb_radix);
  const _scheme = Buffer.from(cell.fragment.substring(idx + 1), "base64").toString("utf-8");
  if (isNaN(handle)) {
    return undefined;
  }
  return {
    handle,
    notebook: cell.with({ scheme: _scheme, fragment: "" }),
  };
}

function generateNotebookCellUri(notebook: Uri, handle: number): Uri {
  const s = handle.toString(nb_radix);
  const p = s.length < nb_lengths.length ? nb_lengths[s.length - 1] : "z";
  const fragment = `${p}${s}s${Buffer.from(notebook.scheme).toString("base64")}`;
  return notebook.with({ scheme: DocumentSchemes.vscodeNotebookCell, fragment });
}
