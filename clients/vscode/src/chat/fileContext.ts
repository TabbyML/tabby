import type { TextEditor, TextDocument } from "vscode";
import type { FileContext } from "tabby-chat-panel";
import type { GitProvider } from "../git/GitProvider";
import { workspace, window, Position, Range, Selection, TextEditorRevealType, Uri, ViewColumn } from "vscode";
import path from "path";
import { getLogger } from "../logger";

const logger = getLogger("FileContext");

export async function getFileContextFromSelection(
  editor: TextEditor,
  gitProvider: GitProvider,
): Promise<FileContext | null> {
  return getFileContext(editor, gitProvider, true);
}

export async function getFileContextFromActiveEditor(
  editor: TextEditor,
  gitProvider: GitProvider,
): Promise<FileContext | null> {
  return getFileContext(editor, gitProvider, true, true);
}

export async function getFileContext(
  editor: TextEditor,
  gitProvider: GitProvider,
  useSelection = false,
  alwaysReturnContext = false,
): Promise<FileContext | null> {
  const uri = editor.document.uri;
  const text = editor.document.getText(useSelection ? editor.selection : undefined);
  const isEmptyText = !text || text.trim().length < 1;
  if (isEmptyText && !alwaysReturnContext) {
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

  if (alwaysReturnContext && isEmptyText) {
    range.start = 0;
    range.end = 0;
  }

  const workspaceFolder = workspace.getWorkspaceFolder(uri);
  const repo = gitProvider.getRepository(uri);
  const gitRemoteUrl = repo ? gitProvider.getDefaultRemoteUrl(repo) : undefined;
  let filePath = uri.toString(true);
  if (repo && gitRemoteUrl) {
    gitProvider.updateRemoteUrlToLocalRoot(gitRemoteUrl, repo.rootUri);
    filePath = path.relative(repo.rootUri.toString(true), filePath);
  } else if (workspaceFolder) {
    filePath = path.relative(workspaceFolder.uri.toString(true), filePath);
  }

  return {
    kind: "file",
    content,
    range,
    filepath: filePath,
    git_url: gitRemoteUrl ?? "",
  };
}

export async function showFileContext(fileContext: FileContext, gitProvider: GitProvider): Promise<void> {
  const document = await openTextDocument(fileContext, gitProvider);
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

async function openTextDocument(fileContext: FileContext, gitProvider: GitProvider): Promise<TextDocument | null> {
  const { filepath: filePath, git_url: gitUrl } = fileContext;
  try {
    // try parse as absolute path
    const absoluteFilepath = Uri.parse(filePath, true);
    if (absoluteFilepath.scheme) {
      return workspace.openTextDocument(absoluteFilepath);
    }
  } catch (err) {
    // Cannot open as absolute path, try to find file in git root
  }

  if (gitUrl && gitUrl.trim().length > 0) {
    const localGitRoot = gitProvider.findLocalRootUriByRemoteUrl(gitUrl);
    if (localGitRoot) {
      try {
        const absoluteFilepath = Uri.joinPath(localGitRoot, filePath);
        return await workspace.openTextDocument(absoluteFilepath);
      } catch (err) {
        // File not found in local git root, try to find file in workspace folders
      }
    }
  }

  for (const root of workspace.workspaceFolders ?? []) {
    const absoluteFilepath = Uri.joinPath(root.uri, filePath);
    try {
      return await workspace.openTextDocument(absoluteFilepath);
    } catch (err) {
      // File not found in workspace folder, try to use findFiles
    }
  }
  logger.info("File not found in workspace folders, trying with findFiles...");

  const files = await workspace.findFiles(filePath, undefined, 1);
  if (files[0]) {
    return workspace.openTextDocument(files[0]);
  }

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
