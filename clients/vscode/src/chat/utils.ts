import path from "path";
import { TextEditor, Position as VSCodePosition, Range as VSCodeRange, Uri, workspace } from "vscode";
import type {
  Filepath,
  Position as ChatPanelPosition,
  LineRange,
  PositionRange,
  Location,
  FilepathInGitRepository,
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
  let uriFilePath = uri.toString(true);
  if (uri.scheme === DocumentSchemes.vscodeNotebookCell) {
    const notebook = parseNotebookCellUri(uri);
    if (notebook) {
      // add fragment `#cell={number}` to filepath
      uriFilePath = uri.with({ scheme: notebook.notebook.scheme, fragment: `cell=${notebook.handle}` }).toString(true);
    }
  }

  const workspaceFolder = workspace.getWorkspaceFolder(uri);

  let repo = gitProvider.getRepository(uri);
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

  return {
    kind: "uri",
    uri: uriFilePath,
  };
}

function isJupyterNotebookFilepath(filepath: Filepath): boolean {
  const _filepath = filepath.kind === "uri" ? filepath.uri : filepath.filepath;
  const extname = path.extname(_filepath);
  return extname.startsWith(".ipynb");
}

export function chatPanelFilepathToLocalUri(filepath: Filepath, gitProvider: GitProvider): Uri | null {
  const isNotebook = isJupyterNotebookFilepath(filepath);

  if (filepath.kind === "uri") {
    try {
      if (isNotebook) {
        const handle = chatPanelFilePathToNotebookCellHandle(filepath.uri);
        if (typeof handle === "number") {
          return generateLocalNotebookCellUri(Uri.parse(filepath.uri), handle);
        }
      }

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
      // handling for Jupyter Notebook (.ipynb) files
      if (isNotebook) {
        return chatPanelFilepathToVscodeNotebookCellUri(localGitRoot, filepath);
      }

      return Uri.joinPath(localGitRoot, filepath.filepath);
    }
  }
  logger.warn(`Invalid filepath params.`, filepath);
  return null;
}

function chatPanelFilepathToVscodeNotebookCellUri(root: Uri, filepath: FilepathInGitRepository): Uri | null {
  if (filepath.kind !== "git") {
    logger.warn(`Invalid filepath params.`, filepath);
    return null;
  }

  const filePathUri = Uri.parse(filepath.filepath);
  const notebookUri = Uri.joinPath(root, filePathUri.path);

  const handle = chatPanelFilePathToNotebookCellHandle(filepath.filepath);
  if (typeof handle === "undefined") {
    logger.warn(`Invalid filepath params.`, filepath);
    return null;
  }
  return generateLocalNotebookCellUri(notebookUri, handle);
}

function chatPanelFilePathToNotebookCellHandle(filepath: string): number | undefined {
  let handle: number | undefined;

  const fileUri = Uri.parse(filepath);
  const fragment = fileUri.fragment;
  const searchParams = new URLSearchParams(fragment);
  if (searchParams.has("cell")) {
    const cellString = searchParams.get("cell")?.toString() || "";
    handle = parseInt(cellString, 10);
  }

  if (typeof handle === "undefined" || isNaN(handle)) {
    return undefined;
  }

  return handle;
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

const nb_lengths = ["W", "X", "Y", "Z", "a", "b", "c", "d", "e", "f"];
const nb_padRegexp = new RegExp(`^[${nb_lengths.join("")}]+`);
const nb_radix = 7;
export function parseNotebookCellUri(cell: Uri): { notebook: Uri; handle: number } | undefined {
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

export function generateLocalNotebookCellUri(notebook: Uri, handle: number): Uri {
  const s = handle.toString(nb_radix);
  const p = s.length < nb_lengths.length ? nb_lengths[s.length - 1] : "z";
  const fragment = `${p}${s}s${Buffer.from(notebook.scheme).toString("base64")}`;
  return notebook.with({ scheme: DocumentSchemes.vscodeNotebookCell, fragment });
}
