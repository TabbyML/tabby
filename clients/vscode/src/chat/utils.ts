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

const logger = getLogger("chat/utils");

export function localUriToChatPanelFilepath(uri: Uri, gitProvider: GitProvider): Filepath {
  const workspaceFolder = workspace.getWorkspaceFolder(uri);

  let repo = gitProvider.getRepository(uri);
  if (!repo && workspaceFolder) {
    repo = gitProvider.getRepository(workspaceFolder.uri);
  }
  const gitRemoteUrl = repo ? gitProvider.getDefaultRemoteUrl(repo) : undefined;
  if (repo && gitRemoteUrl) {
    const uriFilePath =
      uri.scheme === "vscode-notebook-cell" ? uri.with({ scheme: "file" }).toString(true) : uri.toString(true);
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
    uri: uri.toString(true),
  };
}

export function vscodeNoteCellUriToChagePanelRange(uri: Uri) {
  if (uri.scheme !== "vscode-notebook-cell") return undefined;
  const notebook = parseVscodeNotebookCellURI(uri);
  return notebook;
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

  const parsedUrl = new URL(filepath.filepath, "file://");
  const hash = parsedUrl.hash;
  const cleanPath = parsedUrl.pathname;
  return Uri.joinPath(root, cleanPath).with({ scheme: "vscode-notebook-cell", fragment: hash.slice(1) });
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
  } else if ("cellIndex" in location) {
    // FIXME cellIndex?
    return chatPanelLineRangeToVSCodeRange(location as LineRange);
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

export function parseVscodeNotebookCellURI(cell: Uri) {
  if (!cell.scheme) return undefined;
  if (!cell.scheme.startsWith("vscode-notebook-cell")) return undefined;

  const _lengths = ["W", "X", "Y", "Z", "a", "b", "c", "d", "e", "f"];
  const _padRegexp = new RegExp(`^[${_lengths.join("")}]+`);
  const _radix = 7;
  const fragment = cell.fragment.split("#").pop() || "";
  const idx = fragment.indexOf("s");
  if (idx < 0) {
    return undefined;
  }
  const handle = parseInt(fragment.substring(0, idx).replace(_padRegexp, ""), _radix);
  const _scheme = Buffer.from(fragment.substring(idx + 1), "base64").toString("utf-8");

  if (isNaN(handle)) {
    return undefined;
  }
  return {
    handle,
    scheme: _scheme,
  };
}
