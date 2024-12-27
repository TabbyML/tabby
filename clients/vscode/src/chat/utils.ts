import path from "path";
import { Position as VSCodePosition, Range as VSCodeRange, Uri, workspace } from "vscode";
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
import { Schemes } from "./chat";

const logger = getLogger("chat/utils");

export function localUriToChatPanelFilepath(uri: Uri, gitProvider: GitProvider): Filepath {
  const workspaceFolder = workspace.getWorkspaceFolder(uri);

  let repo = gitProvider.getRepository(uri);
  if (!repo && workspaceFolder) {
    repo = gitProvider.getRepository(workspaceFolder.uri);
  }
  const gitRemoteUrl = repo ? gitProvider.getDefaultRemoteUrl(repo) : undefined;
  if (repo && gitRemoteUrl) {
    let uriFilePath: string = uri.toString(true);

    if (uri.scheme === Schemes.vscodeNotebookCell) {
      uriFilePath = localUriToUriFilePath(uri);
    }

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
    uri: localUriToUriFilePath(uri),
  };
}

function localUriToUriFilePath(uri: Uri): string {
  let uriFilePath = uri.toString(true);

  if (uri.scheme === Schemes.vscodeNotebookCell) {
    const notebook = parseNotebookCellUri(uri);
    if (notebook) {
      // add fragment `#cell={number}` to filepath
      uriFilePath = uri.with({ scheme: notebook.notebook.scheme, fragment: `cell=${notebook.handle}` }).toString(true);
    }
  }
  return uriFilePath;
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
      const extname = path.extname(filepath.filepath);

      // handling for Jupyter Notebook (.ipynb) files
      if (extname.startsWith(".ipynb")) {
        return chatPanelFilepathToVscodeNotebookCellUri(localGitRoot, filepath);
      }

      return Uri.joinPath(localGitRoot, filepath.filepath);
    }
  }
  logger.warn(`Invalid filepath params.`, filepath);
  return null;
}

function chatPanelFilepathToVscodeNotebookCellUri(root: Uri, filepath: FilepathInGitRepository) {
  if (filepath.kind !== "git") {
    logger.warn(`Invalid filepath params.`, filepath);
    return null;
  }

  const uri = Uri.joinPath(root, filepath.filepath);
  let handle: number | undefined;
  const fragment = uri.fragment;
  if (fragment.startsWith("cell=")) {
    const handleStr = fragment.slice("cell=".length);
    const _handle = parseInt(handleStr, 10);
    if (isNaN(_handle)) {
      return uri;
    }
  }

  if (typeof handle === "undefined") {
    logger.warn(`Invalid handle in filepath.`, filepath.filepath);
    return uri;
  }

  const cellUri = generateNotebookCellUri(uri, handle);
  return cellUri;
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
  if (cell.scheme !== Schemes.vscodeNotebookCell) {
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

export function generateNotebookCellUri(notebook: Uri, handle: number): Uri {
  const s = handle.toString(nb_radix);
  const p = s.length < nb_lengths.length ? nb_lengths[s.length - 1] : "z";
  const fragment = `${p}${s}s${Buffer.from(notebook.scheme).toString("base64")}`;
  return notebook.with({ scheme: Schemes.vscodeNotebookCell, fragment });
}
