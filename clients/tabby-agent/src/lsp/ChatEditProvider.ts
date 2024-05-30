import { Range, Location, Connection, CancellationToken } from "vscode-languageserver";
import {
  ChatEditToken,
  ChatEditRequest,
  ChatEditParams,
  ChatEditResolveRequest,
  ChatEditResolveParams,
  ChatFeatureNotAvailableError,
  ChatEditDocumentTooLongError,
  ChatEditCommandTooLongError,
  ChatEditMutexError,
} from "./protocol";
import { TextDocuments } from "./TextDocuments";
import { TextDocument } from "vscode-languageserver-textdocument";
import { Readable } from "node:stream";
import * as RandomString from "randomstring";
import * as Diff from "diff";
import { TabbyAgent } from "../TabbyAgent";
import { isBlank } from "../utils";

export type Edit = {
  id: ChatEditToken;
  location: Location;
  languageId: string;
  originalText: string;
  editedRange: Range;
  editedText: string;
  comments: string;
  buffer: string;
  state: "editing" | "stopped" | "completed";
};

export class ChatEditProvider {
  private currentEdit: Edit | null = null;
  private mutexAbortController: AbortController | null = null;

  constructor(
    private readonly connection: Connection,
    private readonly documents: TextDocuments<TextDocument>,
    private readonly agent: TabbyAgent,
  ) {
    this.connection.onRequest(ChatEditRequest.type, async (params, token) => {
      return this.provideEdit(params, token);
    });
    this.connection.onRequest(ChatEditResolveRequest.type, async (params) => {
      return this.resolveEdit(params);
    });
  }

  isCurrentEdit(id: ChatEditToken): boolean {
    return this.currentEdit?.id === id;
  }

  async provideEdit(params: ChatEditParams, token: CancellationToken): Promise<ChatEditToken | null> {
    if (params.format !== "previewChanges") {
      return null;
    }
    const document = this.documents.get(params.location.uri);
    if (!document) {
      return null;
    }
    if (!this.agent.getServerHealthState()?.chat_model) {
      throw {
        name: "ChatFeatureNotAvailableError",
        message: "Chat feature not available",
      } as ChatFeatureNotAvailableError;
    }
    const { documentMaxChars, commandMaxChars, responseSplitter } = this.agent.getConfig().experimentalChat.edit;
    const documentText = document.getText(params.location.range);
    if (documentText.length > documentMaxChars) {
      throw { name: "ChatEditDocumentTooLongError", message: "Document too long" } as ChatEditDocumentTooLongError;
    }
    if (params.command.length > commandMaxChars) {
      throw { name: "ChatEditCommandTooLongError", message: "Command too long" } as ChatEditCommandTooLongError;
    }
    if (this.mutexAbortController) {
      throw {
        name: "ChatEditMutexError",
        message: "Another smart edit is already in progress",
      } as ChatEditMutexError;
    }
    this.mutexAbortController = new AbortController();
    token.onCancellationRequested(() => this.mutexAbortController?.abort());

    const readableStream = await this.agent.provideChatEdit(documentText, params.command, {
      signal: this.mutexAbortController.signal,
    });

    const editId = "tabby-" + RandomString.generate({ length: 6, charset: "alphanumeric" });
    this.currentEdit = {
      id: editId,
      location: params.location,
      languageId: document.languageId,
      originalText: documentText,
      editedRange: params.location.range,
      editedText: "",
      comments: "",
      buffer: "",
      state: "editing",
    };
    if (!readableStream) {
      return null;
    }
    await this.readResponseStream(readableStream, responseSplitter);
    return editId;
  }

  async stopEdit(id: ChatEditToken): Promise<void> {
    if (this.isCurrentEdit(id)) {
      this.mutexAbortController?.abort();
    }
  }

  async resolveEdit(params: ChatEditResolveParams): Promise<boolean> {
    const document = this.documents.get(params.location.uri);
    if (!document) {
      return false;
    }
    const header = document.getText({
      start: {
        line: params.location.range.start.line,
        character: 0,
      },
      end: {
        line: params.location.range.start.line + 1,
        character: 0,
      },
    });
    const match = /^<<<<<<<.+(<.*>)\[(tabby-[0-9|a-z|A-Z]{6})\]/g.exec(header);
    const markers = match?.[1];
    if (!match || !markers) {
      return false;
    }
    const previewRange = {
      start: {
        line: params.location.range.start.line,
        character: 0,
      },
      end: {
        line: params.location.range.start.line + markers.length,
        character: 0,
      },
    };
    const previewText = document.getText(previewRange);
    const previewLines = previewText.split("\n");
    const lines: string[] = [];
    previewLines.forEach((line, lineIndex) => {
      const marker = markers[lineIndex];
      if (!marker) {
        return;
      }
      if (params.action === "accept") {
        if ([".", "|", "=", "+"].includes(marker)) {
          lines.push(line);
        }
      }
      if (params.action === "discard") {
        if ([".", "=", "-"].includes(marker)) {
          lines.push(line);
        }
      }
    });
    await this.connection.workspace.applyEdit({
      edit: {
        changes: {
          [params.location.uri]: [
            {
              range: previewRange,
              newText: lines.join("\n") + "\n",
            },
          ],
        },
      },
    });
    return true;
  }

  private async readResponseStream(stream: Readable, responseSplitter: string): Promise<void> {
    const finalize = async (state: "completed" | "stopped") => {
      if (this.currentEdit) {
        const edit = this.currentEdit;
        edit.state = state;
        const editedLines = this.generateChangesPreview(edit);
        await this.connection.workspace.applyEdit({
          edit: {
            changes: {
              [edit.location.uri]: [
                {
                  range: edit.editedRange,
                  newText: editedLines.join("\n") + "\n",
                },
              ],
            },
          },
        });
      }
      this.currentEdit = null;
      this.mutexAbortController = null;
    };
    try {
      let currentBlock = 0;
      for await (const delta of stream) {
        if (!this.currentEdit || !this.mutexAbortController || this.mutexAbortController.signal.aborted) {
          break;
        }
        let changed = false;
        const edit = this.currentEdit;
        edit.buffer += delta;
        const lines = edit.buffer.split("\n");
        edit.buffer = "";
        for (let lineIndex = 0; lineIndex < lines.length - 1; lineIndex++) {
          const line = lines[lineIndex];
          if (line === undefined) {
            break;
          }
          if (line.trim().startsWith(responseSplitter)) {
            currentBlock++;
          } else {
            if (currentBlock === 1) {
              edit.editedText += line + "\n";
              changed = true;
            } else if (currentBlock === 2) {
              edit.comments += line + "\n";
              changed = true;
            }
          }
        }
        const lastLine = lines[lines.length - 1];
        if (lastLine) {
          if (responseSplitter.startsWith(lastLine.trim()) || lastLine.trim().startsWith(responseSplitter)) {
            // seems to be a splitter, keep it
            edit.buffer = lastLine;
          } else {
            if (currentBlock === 1) {
              edit.editedText += lastLine;
              changed = true;
            } else if (currentBlock === 2) {
              edit.comments += lastLine;
              changed = true;
            }
          }
        }
        if (changed) {
          const editedLines = this.generateChangesPreview(edit);
          await this.connection.workspace.applyEdit({
            edit: {
              changes: {
                [edit.location.uri]: [
                  {
                    range: edit.editedRange,
                    newText: editedLines.join("\n") + "\n",
                  },
                ],
              },
            },
          });
          edit.editedRange = {
            start: {
              line: edit.editedRange.start.line,
              character: 0,
            },
            end: {
              line: edit.editedRange.start.line + editedLines.length,
              character: 0,
            },
          };
        }
      }
    } catch (error) {
      await finalize("stopped");
      throw error;
    }
    await finalize("completed");
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
  private generateChangesPreview(edit: Edit): string[] {
    const lines: string[] = [];
    let markers = "";
    // header
    let stateDescription = "Editing in progress";
    if (edit.state === "stopped") {
      stateDescription = "Editing stopped";
    } else if (edit.state == "completed") {
      stateDescription = "Editing completed";
    }
    lines.push(`<<<<<<< ${stateDescription} {{markers}}[${edit.id}]`);
    markers += "<";
    // comments: split by new line or 80 chars
    const commentLines = edit.comments
      .trim()
      .split(/\n|(.{1,80})(?:\s|$)/g)
      .filter((input) => !isBlank(input));
    const commentPrefix = this.getCommentPrefix(edit.languageId);
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
        pushDiffValue(lastDiff.value, "|");
      }
      while (lineIndex < diffs.length - inProgressChunk) {
        const diff = diffs[lineIndex];
        if (!diff) {
          break;
        }
        pushDiffValue(diff.value, ".");
        lineIndex++;
      }
    }
    // footer
    lines.push(`>>>>>>> {{markers}}[${edit.id}]`);
    markers += ">";
    // replace markers
    lines[0] = lines[0]!.replace("{{markers}}", markers);
    lines[lines.length - 1] = lines[lines.length - 1]!.replace("{{markers}}", markers);
    return lines;
  }

  // FIXME: improve this
  private getCommentPrefix(languageId: string) {
    if (["plaintext", "markdown"].includes(languageId)) {
      return "";
    }
    if (["python", "ruby"].includes(languageId)) {
      return "#";
    }
    if (["c", "cpp", "java", "javascript", "typescript", "go", "rust", "swift", "kotlin"].includes(languageId)) {
      return "//";
    }
    return "";
  }
}
