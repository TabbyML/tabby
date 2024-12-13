import type { TextEditor, TextDocument } from "vscode";
import type { FileContext } from "tabby-chat-panel";
import type { GitProvider } from "../git/GitProvider";
import { workspace, window, Position, Range, Selection, TextEditorRevealType, Uri, ViewColumn } from "vscode";
import path from "path";
import { getLogger } from "../logger";

const logger = getLogger("FileContext");

export interface FilePathParams {
  filePath: string;
  gitRemoteUrl?: string;
}

export async function getFileContextFromSelection(
  editor: TextEditor,
  gitProvider: GitProvider,
): Promise<FileContext | null> {
  return getFileContext(editor, gitProvider, true);
}

export async function getFileContext(
  editor: TextEditor,
  gitProvider: GitProvider,
  useSelection = false,
): Promise<FileContext | null> {
  const text = editor.document.getText(useSelection ? editor.selection : undefined);
  if (!text || text.trim().length < 1) {
    return null;
  }
  const content = useSelection ? alignIndent(text) : text;
  const range = useSelection
    ? {
        start: editor.selection.start.line + 1,
        end: editor.selection.end.line + 1,
      }
    : {
        start: 1,
        end: editor.document.lineCount,
      };

  const filePathParams = await buildFilePathParams(editor.document.uri, gitProvider);

  return {
    kind: "file",
    content,
    range,
    filepath: filePathParams.filePath,
    git_url: filePathParams.gitRemoteUrl ?? "",
  };
}

export async function showFileContext(fileContext: FileContext, gitProvider: GitProvider): Promise<void> {
  const document = await openTextDocument(
    {
      filePath: fileContext.filepath,
      gitRemoteUrl: fileContext.git_url,
    },
    gitProvider,
  );
  if (!document) {
    throw new Error(`File not found: ${fileContext.filepath}`);
  }

  const editor = await window.showTextDocument(document, {
    viewColumn: ViewColumn.Active,
    preview: false,
    preserveFocus: true,
  });

  // Move the cursor to the specified line
  const start = new Position(Math.max(0, fileContext.range.start - 1), 0);
  const end = new Position(fileContext.range.end, 0);
  editor.selection = new Selection(start, end);
  editor.revealRange(new Range(start, end), TextEditorRevealType.InCenter);
}

export async function buildFilePathParams(uri: Uri, gitProvider: GitProvider): Promise<FilePathParams> {
  const workspaceFolder =
    workspace.getWorkspaceFolder(uri) ?? (uri.scheme === "untitled" ? workspace.workspaceFolders?.[0] : undefined);
  const repo =
    gitProvider.getRepository(uri) ?? (workspaceFolder ? gitProvider.getRepository(workspaceFolder.uri) : undefined);
  const gitRemoteUrl = repo ? gitProvider.getDefaultRemoteUrl(repo) : undefined;
  let filePath = uri.toString(true);
  if (repo && gitRemoteUrl) {
    const relativeFilePath = path.relative(repo.rootUri.toString(true), filePath);
    if (!relativeFilePath.startsWith("..")) {
      filePath = relativeFilePath;
    }
  } else if (workspaceFolder) {
    const relativeFilePath = path.relative(workspaceFolder.uri.toString(true), filePath);
    if (!relativeFilePath.startsWith("..")) {
      filePath = relativeFilePath;
    }
  }
  return {
    filePath,
    gitRemoteUrl,
  };
}

export async function openTextDocument(
  filePathParams: FilePathParams,
  gitProvider: GitProvider,
): Promise<TextDocument | null> {
  const { filePath, gitRemoteUrl } = filePathParams;

  // Try parse as absolute path
  try {
    const absoluteFilepath = Uri.parse(filePath, true);
    if (absoluteFilepath.scheme) {
      return workspace.openTextDocument(absoluteFilepath);
    }
  } catch (err) {
    // ignore
  }

  // Try find file in provided git repository
  if (gitRemoteUrl && gitRemoteUrl.trim().length > 0) {
    const localGitRoot = gitProvider.findLocalRootUriByRemoteUrl(gitRemoteUrl);
    if (localGitRoot) {
      try {
        const absoluteFilepath = Uri.joinPath(localGitRoot, filePath);
        return await workspace.openTextDocument(absoluteFilepath);
      } catch (err) {
        // ignore
      }
    }
  }

  for (const root of workspace.workspaceFolders ?? []) {
    // Try find file in workspace folder
    const absoluteFilepath = Uri.joinPath(root.uri, filePath);
    try {
      return await workspace.openTextDocument(absoluteFilepath);
    } catch (err) {
      // ignore
    }

    // Try find file in git repository of workspace folder
    const localGitRoot = gitProvider.getRepository(root.uri)?.rootUri;
    if (localGitRoot) {
      try {
        const absoluteFilepath = Uri.joinPath(localGitRoot, filePath);
        return await workspace.openTextDocument(absoluteFilepath);
      } catch (err) {
        // ignore
      }
    }
  }

  // Try find file in workspace folders using workspace.findFiles
  logger.info("File not found in workspace folders, trying with findFiles...");
  const files = await workspace.findFiles(filePath, undefined, 1);
  if (files[0]) {
    try {
      return await workspace.openTextDocument(files[0]);
    } catch (err) {
      // ignore
    }
  }

  logger.warn(`File not found: ${filePath}`);
  return null;
}

function alignIndent(text: string): string {
  const lines = text.split("\n");
  const subsequentLines = lines.slice(1);

  // Determine the minimum indent for subsequent lines
  const minIndent = subsequentLines.reduce((min, line) => {
    const match = line.match(/^(\s*)/);
    const indent = match ? match[0].length : 0;
    return line.trim() ? Math.min(min, indent) : min;
  }, Infinity);

  // Remove the minimum indent
  const adjustedLines = lines.slice(1).map((line) => line.slice(minIndent));

  return [lines[0]?.trim(), ...adjustedLines].join("\n");
}
