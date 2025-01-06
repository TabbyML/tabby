import type { TextEditor } from "vscode";
import type { EditorContext } from "tabby-chat-panel";
import type { GitProvider } from "../git/GitProvider";
import { localUriToChatPanelFilepath } from "./utils";

export async function getFileContextFromSelection(
  editor: TextEditor,
  gitProvider: GitProvider,
): Promise<EditorContext | null> {
  return getFileContext(editor, gitProvider, true);
}

export async function getFileContext(
  editor: TextEditor,
  gitProvider: GitProvider,
  useSelection = false,
): Promise<EditorContext | null> {
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
    : undefined;

  const filepath = localUriToChatPanelFilepath(editor.document.uri, gitProvider);

  return {
    kind: "file",
    filepath,
    range,
    content,
  };
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
