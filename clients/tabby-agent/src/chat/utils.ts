//chat related utils functions

import { Readable } from "stream";
import {
  Range,
  Location,
  ShowDocumentParams,
  ShowDocumentRequest,
  WorkspaceEdit,
} from "vscode-languageserver-protocol";
import { Connection } from "vscode-languageserver";
import * as Diff from "diff";
import { ApplyWorkspaceEditParams, ApplyWorkspaceEditRequest } from "../protocol";
import { isBlank } from "../utils/string";

export type Edit = {
  id: string;
  location: Location;
  languageId: string;
  originalText: string;
  editedRange: Range;
  editedText: string;
  comments: string;
  buffer: string;
  state: "editing" | "stopped" | "completed";
};

export async function readResponseStream(
  stream: Readable,
  connection: Connection,
  currentEdit: Edit | undefined,
  mutexAbortController: AbortController | undefined,
  resetEditAndMutexAbortController: () => void,
  responseDocumentTag: string[],
  responseCommentTag?: string[],
): Promise<void> {
  const applyEdit = async (edit: Edit, isFirst: boolean = false, isLast: boolean = false) => {
    if (isFirst) {
      const workspaceEdit: WorkspaceEdit = {
        changes: {
          [edit.location.uri]: [
            {
              range: {
                start: { line: edit.editedRange.start.line, character: 0 },
                end: { line: edit.editedRange.start.line, character: 0 },
              },
              newText: `<<<<<<< ${edit.id}\n`,
            },
          ],
        },
      };

      await applyWorkspaceEdit(
        {
          edit: workspaceEdit,
          options: {
            undoStopBefore: true,
            undoStopAfter: false,
          },
        },
        connection,
      );

      edit.editedRange = {
        start: { line: edit.editedRange.start.line + 1, character: 0 },
        end: { line: edit.editedRange.end.line + 1, character: 0 },
      };
    }

    const editedLines = generateChangesPreview(edit);
    const workspaceEdit: WorkspaceEdit = {
      changes: {
        [edit.location.uri]: [
          {
            range: edit.editedRange,
            newText: editedLines.join("\n") + "\n",
          },
        ],
      },
    };

    await applyWorkspaceEdit(
      {
        edit: workspaceEdit,
        options: {
          undoStopBefore: false,
          undoStopAfter: isLast,
        },
      },
      connection,
    );

    edit.editedRange = {
      start: { line: edit.editedRange.start.line, character: 0 },
      end: { line: edit.editedRange.start.line + editedLines.length, character: 0 },
    };
  };

  const processBuffer = (edit: Edit, inTag: "document" | "comment", openTag: string, closeTag: string) => {
    if (edit.buffer.startsWith(openTag)) {
      edit.buffer = edit.buffer.substring(openTag.length);
    }

    const reg = createCloseTagMatcher(closeTag);
    const match = reg.exec(edit.buffer);
    if (!match) {
      edit[inTag === "document" ? "editedText" : "comments"] += edit.buffer;
      edit.buffer = "";
    } else {
      edit[inTag === "document" ? "editedText" : "comments"] += edit.buffer.substring(0, match.index);
      edit.buffer = edit.buffer.substring(match.index);
      return match[0] === closeTag ? false : inTag;
    }
    return inTag;
  };
  const findOpenTag = (
    buffer: string,
    responseDocumentTag: string[],
    responseCommentTag?: string[],
  ): "document" | "comment" | false => {
    const openTags = [responseDocumentTag[0], responseCommentTag?.[0]].filter(Boolean);
    if (openTags.length < 1) return false;

    const reg = new RegExp(openTags.join("|"), "g");
    const match = reg.exec(buffer);
    if (match && match[0]) {
      if (match[0] === responseDocumentTag[0]) {
        return "document";
      } else if (match[0] === responseCommentTag?.[0]) {
        return "comment";
      }
    }
    return false;
  };

  try {
    if (!currentEdit) {
      throw new Error("No current edit");
    }

    let inTag: "document" | "comment" | false = false;

    // Insert the first line as early as possible so codelens can be shown
    await applyEdit(currentEdit, true, false);

    for await (const item of stream) {
      if (!mutexAbortController || mutexAbortController.signal.aborted) {
        break;
      }
      const delta = typeof item === "string" ? item : "";
      const edit = currentEdit;
      edit.buffer += delta;

      if (!inTag) {
        inTag = findOpenTag(edit.buffer, responseDocumentTag, responseCommentTag);
      }

      if (inTag) {
        const openTag = inTag === "document" ? responseDocumentTag[0] : responseCommentTag?.[0];
        const closeTag = inTag === "document" ? responseDocumentTag[1] : responseCommentTag?.[1];
        if (!closeTag || !openTag) break;
        inTag = processBuffer(edit, inTag, openTag, closeTag);
        if (delta.includes("\n")) {
          await applyEdit(edit, false, false);
        }
      }
    }

    if (currentEdit) {
      currentEdit.state = "completed";
      await applyEdit(currentEdit, false, true);
    }
  } catch (error) {
    if (currentEdit) {
      currentEdit.state = "stopped";
      await applyEdit(currentEdit, false, true);
    }
    if (!(error instanceof TypeError && error.message.startsWith("terminated"))) {
      throw error;
    }
  } finally {
    resetEditAndMutexAbortController();
  }
}

export async function applyWorkspaceEdit(
  params: ApplyWorkspaceEditParams,
  lspConnection: Connection,
): Promise<boolean> {
  if (!lspConnection) {
    return false;
  }
  try {
    // FIXME(Sma1lboy): adding client capabilities to indicate if client support this method rather than try-catch
    const result = await lspConnection.sendRequest(ApplyWorkspaceEditRequest.type, params);
    return result;
  } catch (error) {
    try {
      await lspConnection.workspace.applyEdit({
        edit: params.edit,
        label: params.label,
      });
      return true;
    } catch (fallbackError) {
      return false;
    }
  }
}

export async function showDocument(params: ShowDocumentParams, lspConnection: Connection): Promise<boolean> {
  if (!lspConnection) {
    return false;
  }

  try {
    const result = await lspConnection.sendRequest(ShowDocumentRequest.type, params);
    return result.success;
  } catch (error) {
    return false;
  }
}

// header line
// <<<<<<< Editing by Tabby <.#=+->
// markers:
// [<] header
// [#] comments
// [.] waiting
// [|] in progress
// [=] unchanged
// [+] inserted
// [-] deleted
// [>] footer
// footer line
// >>>>>>> End of changes
export function generateChangesPreview(edit: Edit): string[] {
  const lines: string[] = [];
  let markers = "";
  // lines.push(`<<<<<<< ${stateDescription} {{markers}}[${edit.id}]`);
  markers += "[";
  // comments: split by new line or 80 chars
  const commentLines = edit.comments
    .trim()
    .split(/\n|(.{1,80})(?:\s|$)/g)
    .filter((input) => !isBlank(input));
  const commentPrefix = getCommentPrefix(edit.languageId);
  for (const line of commentLines) {
    lines.push(commentPrefix + line);
    markers += "#";
  }
  const pushDiffValue = (diffValue: string, marker: string) => {
    diffValue
      .replace(/\n$/, "")
      .split("\n")
      .forEach((line) => {
        lines.push(line);
        markers += marker;
      });
  };
  // diffs
  const diffs = Diff.diffLines(edit.originalText, edit.editedText);
  if (edit.state === "completed") {
    diffs.forEach((diff) => {
      if (diff.added) {
        pushDiffValue(diff.value, "+");
      } else if (diff.removed) {
        pushDiffValue(diff.value, "-");
      } else {
        pushDiffValue(diff.value, "=");
      }
    });
  } else {
    let inProgressChunk = 0;
    const lastDiff = diffs[diffs.length - 1];
    if (lastDiff && lastDiff.added) {
      inProgressChunk = 1;
    }
    let waitingChunks = 0;
    for (let i = diffs.length - inProgressChunk - 1; i >= 0; i--) {
      if (diffs[i]?.removed) {
        waitingChunks++;
      } else {
        break;
      }
    }
    let lineIndex = 0;
    while (lineIndex < diffs.length - inProgressChunk - waitingChunks) {
      const diff = diffs[lineIndex];
      if (!diff) {
        break;
      }
      if (diff.added) {
        pushDiffValue(diff.value, "+");
      } else if (diff.removed) {
        pushDiffValue(diff.value, "-");
      } else {
        pushDiffValue(diff.value, "=");
      }
      lineIndex++;
    }
    if (inProgressChunk && lastDiff) {
      if (edit.state === "stopped") {
        pushDiffValue(lastDiff.value, "+");
      } else {
        pushDiffValue(lastDiff.value, "|");
      }
    }
    while (lineIndex < diffs.length - inProgressChunk) {
      const diff = diffs[lineIndex];
      if (!diff) {
        break;
      }
      if (edit.state === "stopped") {
        pushDiffValue(diff.value, "=");
      } else {
        pushDiffValue(diff.value, ".");
      }
      lineIndex++;
    }
  }
  // footer
  lines.push(`>>>>>>> ${edit.id} {{markers}}`);
  markers += "]";
  // replace markers
  // lines[0] = lines[0]!.replace("{{markers}}", markers);
  lines[lines.length - 1] = lines[lines.length - 1]!.replace("{{markers}}", markers);
  return lines;
}

export function createCloseTagMatcher(tag: string): RegExp {
  let reg = `${tag}`;
  for (let length = tag.length - 1; length > 0; length--) {
    reg += "|" + tag.substring(0, length) + "$";
  }
  return new RegExp(reg, "g");
}

// FIXME: improve this
export function getCommentPrefix(languageId: string) {
  if (["plaintext", "markdown"].includes(languageId)) {
    return "";
  }
  if (["python", "ruby"].includes(languageId)) {
    return "#";
  }
  if (
    [
      "c",
      "cpp",
      "java",
      "javascript",
      "typescript",
      "javascriptreact",
      "typescriptreact",
      "go",
      "rust",
      "swift",
      "kotlin",
    ].includes(languageId)
  ) {
    return "//";
  }
  return "";
}

export function truncateFileContent(content: string, maxLength: number): string {
  if (content.length <= maxLength) {
    return content;
  }

  content = content.slice(0, maxLength);
  const lastNewLine = content.lastIndexOf("\n");
  if (lastNewLine > 0) {
    content = content.slice(0, lastNewLine + 1);
  }

  return content;
}
