import type { TextEditor } from "vscode";
import dedent from "dedent";
import type { EditorFileContext } from "tabby-chat-panel";
import type { GitProvider } from "../git/GitProvider";
import { localUriToChatPanelFilepath, vscodeRangeToChatPanelPositionRange } from "./utils";

// default: use selection if available, otherwise use the whole file
// selection: use selection if available, otherwise return null
// file: use the whole file
export type RangeStrategy = "default" | "selection" | "file";

export async function getEditorContext(
  editor: TextEditor,
  gitProvider: GitProvider,
  rangeStrategy: RangeStrategy = "default",
): Promise<EditorFileContext | null> {
  let range = rangeStrategy !== "file" ? editor.selection : undefined;
  let text = !range?.isEmpty ? editor.document.getText(range) : "";
  if (!text || text.trim().length < 1) {
    if (rangeStrategy === "selection") {
      return null;
    } else if (range !== undefined) {
      range = undefined;
      text = editor.document.getText();
    }
  }
  const content = range !== undefined ? dedent(text) : text;

  return {
    kind: "file",
    filepath: localUriToChatPanelFilepath(editor.document.uri, gitProvider),
    range: range ? vscodeRangeToChatPanelPositionRange(range) : undefined,
    content,
  };
}
