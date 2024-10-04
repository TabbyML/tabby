import { EditorParams } from "tabby-agent";
import { TextEditor, window } from "vscode";

export class FileTrackerProvider {
  collectVisibleEditors(exceptActiveEditor = false, activeEditor?: TextEditor): EditorParams[] {
    let editors = window.visibleTextEditors
      .filter((e) => e.document.fileName.startsWith("/"))
      .map((editor) => {
        if (!editor.visibleRanges[0]) {
          return null;
        }
        return {
          uri: editor.document.uri.toString(),
          visibleRange: {
            start: {
              line: editor.visibleRanges[0].start.line,
              character: editor.visibleRanges[0].start.character,
            },
            end: {
              line: editor.visibleRanges[0].end.line,
              character: editor.visibleRanges[0].end.character,
            },
          },
        } as EditorParams;
      })
      .filter((e): e is EditorParams => e !== null)
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
            e.visibleRange.start.line !== range.start.line ||
            e.visibleRange.start.character !== range.start.character ||
            e.visibleRange.end.line !== range.end.line ||
            e.visibleRange.end.character !== range.end.character,
        );
      }
    }
    return editors;
  }
  collectActiveEditor(): EditorParams | undefined {
    const activeEditor = window.activeTextEditor;
    //only return TextDocument editor
    if (!activeEditor || !activeEditor.visibleRanges[0] || !activeEditor.document.fileName.startsWith("/")) {
      return undefined;
    }
    return {
      uri: activeEditor.document.uri.toString(),
      visibleRange: {
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
}
