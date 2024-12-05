import { TextEditor, window } from "vscode";
import { Location } from "vscode-languageclient";

export function collectVisibleEditors(exceptActiveEditor = false, activeEditor?: TextEditor): Location[] {
  let editors = window.visibleTextEditors
    .filter((e) => e.document.fileName.startsWith("/"))
    .map((editor) => {
      if (!editor.visibleRanges[0]) {
        return null;
      }
      return {
        uri: editor.document.uri.toString(),
        range: {
          start: {
            line: editor.visibleRanges[0].start.line,
            character: editor.visibleRanges[0].start.character,
          },
          end: {
            line: editor.visibleRanges[0].end.line,
            character: editor.visibleRanges[0].end.character,
          },
        },
      } as Location;
    })
    .filter((e): e is Location => e !== null)
    .sort((a, b) =>
      a.uri === window.activeTextEditor?.document.uri.toString()
        ? -1
        : b.uri === window.activeTextEditor?.document.uri.toString()
          ? 1
          : 0,
    );
  if (exceptActiveEditor) {
    if (activeEditor && activeEditor.visibleRanges[0]) {
      const range = activeEditor.visibleRanges[0];
      editors = editors.filter(
        (e) =>
          e.uri !== activeEditor.document.uri.toString() ||
          e.range.start.line !== range.start.line ||
          e.range.start.character !== range.start.character ||
          e.range.end.line !== range.end.line ||
          e.range.end.character !== range.end.character,
      );
    }
  }
  return editors;
}
export function collectActiveEditor(): Location | undefined {
  const activeEditor = window.activeTextEditor;
  //only return TextDocument editor
  if (!activeEditor || !activeEditor.visibleRanges[0] || !activeEditor.document.fileName.startsWith("/")) {
    return undefined;
  }
  return {
    uri: activeEditor.document.uri.toString(),
    range: {
      start: {
        line: activeEditor.visibleRanges[0].start.line,
        character: activeEditor.visibleRanges[0].start.character,
      },
      end: {
        line: activeEditor.visibleRanges[0].end.line,
        character: activeEditor.visibleRanges[0].end.character,
      },
    },
  };
}
