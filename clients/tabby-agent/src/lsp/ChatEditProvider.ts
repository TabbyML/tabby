import { Range, Location, Connection, CancellationToken, WorkspaceEdit } from "vscode-languageserver";
import {
  ChatEditToken,
  ChatEditRequest,
  ChatEditParams,
  ChatEditResolveRequest,
  ChatEditResolveParams,
  ChatEditCommandRequest,
  ChatEditCommandParams,
  ChatEditCommand,
  ChatFeatureNotAvailableError,
  ChatEditDocumentTooLongError,
  ChatEditCommandTooLongError,
  ChatEditMutexError,
  ApplyWorkspaceEditRequest,
  ApplyWorkspaceEditParams,
  ChatNLOutlinesRequest,
  ChatNLOutlinesParams,
  ChatEditNoDocumentSelectedError,
  ChatNLOutlinesSync,
} from "./protocol";
import { TextDocuments } from "./TextDocuments";
import { TextDocument } from "vscode-languageserver-textdocument";
import { Readable } from "readable-stream";
import cryptoRandomString from "crypto-random-string";
import * as Diff from "diff";
import { TabbyAgent } from "../TabbyAgent";
import { isEmptyRange } from "../utils/range";
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

export interface Outline {
  line: number;
  text: string;
}
export interface Outlines {
  uri: string;
  outlines: Outline[];
}

export class ChatEditProvider {
  private currentEdit: Edit | null = null;
  private mutexAbortController: AbortController | null = null;
  private currentOutlines: Outlines | null = null;

  constructor(
    private readonly connection: Connection,
    private readonly documents: TextDocuments<TextDocument>,
    private readonly agent: TabbyAgent,
  ) {
    this.connection.onRequest(ChatEditCommandRequest.type, async (params) => {
      return this.provideEditCommands(params);
    });
    this.connection.onRequest(ChatEditRequest.type, async (params, token) => {
      return this.provideEdit(params, token);
    });
    this.connection.onRequest(ChatEditResolveRequest.type, async (params) => {
      return this.resolveEdit(params);
    });
    this.connection.onRequest(ChatNLOutlinesRequest.type, async (params, token) => {
      return this.provideNLOutlinesGenerate(params, token);
    });
  }

  isCurrentEdit(id: ChatEditToken): boolean {
    return this.currentEdit?.id === id;
  }

  async provideEditCommands(params: ChatEditCommandParams): Promise<ChatEditCommand[]> {
    const commands = this.agent.getConfig().chat.edit.presetCommands;
    const result: ChatEditCommand[] = [];
    const document = this.documents.get(params.location.uri);

    result.push(
      ...Object.entries(commands)
        .filter(([_command, commandConfig]) => {
          for (const [filterKey, filterValue] of Object.entries(commandConfig.filters)) {
            if (filterValue) {
              switch (filterKey) {
                case "languageIdIn":
                  if (document && !filterValue.split(",").includes(document.languageId)) {
                    return false;
                  }
                  break;
                case "languageIdNotIn":
                  if (document && filterValue.split(",").includes(document.languageId)) {
                    return false;
                  }
                  break;
                default:
                  break;
              }
            }
          }
          return true;
        })
        .map(([command, commandConfig]) => {
          return {
            label: commandConfig.label,
            command,
            source: "preset",
          } as ChatEditCommand;
        }),
    );

    return result;
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
    const config = this.agent.getConfig().chat;
    if (params.command.length > config.edit.commandMaxChars) {
      throw { name: "ChatEditCommandTooLongError", message: "Command too long" } as ChatEditCommandTooLongError;
    }

    let insertMode: boolean = isEmptyRange(params.location.range);
    const presetCommand = /^\/\w+\b/g.exec(params.command)?.[0];
    if (presetCommand) {
      insertMode = config.edit.presetCommands[presetCommand]?.kind === "insert";
    }

    const documentText = document.getText();
    const selection = {
      start: document.offsetAt(params.location.range.start),
      end: document.offsetAt(params.location.range.end),
    };
    const selectedDocumentText = documentText.substring(selection.start, selection.end);
    if (selection.end - selection.start > config.edit.documentMaxChars) {
      throw { name: "ChatEditDocumentTooLongError", message: "Document too long" } as ChatEditDocumentTooLongError;
    }
    if (this.mutexAbortController) {
      throw {
        name: "ChatEditMutexError",
        message: "Another smart edit is already in progress",
      } as ChatEditMutexError;
    }
    this.mutexAbortController = new AbortController();
    token.onCancellationRequested(() => this.mutexAbortController?.abort());

    const readableStream = await this.agent.provideChatEdit(
      documentText,
      selection,
      params.location.uri,
      insertMode,
      params.command,
      document.languageId,
      {
        signal: this.mutexAbortController.signal,
      },
    );

    const editId = "tabby-" + cryptoRandomString({ length: 6, type: "alphanumeric" });
    this.currentEdit = {
      id: editId,
      location: params.location,
      languageId: document.languageId,
      originalText: selectedDocumentText,
      editedRange: insertMode
        ? { start: params.location.range.end, end: params.location.range.end }
        : params.location.range,
      editedText: "",
      comments: "",
      buffer: "",
      state: "editing",
    };
    if (!readableStream) {
      return null;
    }
    await this.readResponseStream(readableStream, config.edit.responseDocumentTag, config.edit.responseCommentTag);
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

    await this.applyWorkspaceEdit({
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
      options: {
        undoStopBefore: false,
        undoStopAfter: false,
      },
    });
    return true;
  }

  private async readResponseStream(
    stream: Readable,
    responseDocumentTag: string[],
    responseCommentTag?: string[],
  ): Promise<void> {
    const applyEdit = async (edit: Edit, isFirst: boolean = false, isLast: boolean = false) => {
      const editedLines = this.generateChangesPreview(edit);
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

      await this.applyWorkspaceEdit({
        edit: workspaceEdit,
        options: {
          undoStopBefore: isFirst,
          undoStopAfter: isLast,
        },
      });

      edit.editedRange = {
        start: { line: edit.editedRange.start.line, character: 0 },
        end: { line: edit.editedRange.start.line + editedLines.length, character: 0 },
      };
    };

    const processBuffer = (edit: Edit, inTag: "document" | "comment", openTag: string, closeTag: string) => {
      if (edit.buffer.startsWith(openTag)) {
        edit.buffer = edit.buffer.substring(openTag.length);
      }

      const reg = this.createCloseTagMatcher(closeTag);
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
      let inTag: "document" | "comment" | false = false;
      let isFirstEdit = true;

      for await (const delta of stream) {
        if (!this.currentEdit || !this.mutexAbortController || this.mutexAbortController.signal.aborted) {
          break;
        }

        const edit = this.currentEdit;
        edit.buffer += delta;

        if (!inTag) {
          inTag = findOpenTag(edit.buffer, responseDocumentTag, responseCommentTag);
        }

        if (inTag) {
          const openTag = inTag === "document" ? responseDocumentTag[0] : responseCommentTag?.[0];
          const closeTag = inTag === "document" ? responseDocumentTag[1] : responseCommentTag?.[1];
          if (!closeTag || !openTag) break;
          inTag = processBuffer(edit, inTag, openTag, closeTag);
          await applyEdit(edit, isFirstEdit, false);
          isFirstEdit = false;
        }
      }

      if (this.currentEdit) {
        this.currentEdit.state = "completed";
        await applyEdit(this.currentEdit, false, true);
      }
    } catch (error) {
      if (this.currentEdit) {
        this.currentEdit.state = "stopped";
        await applyEdit(this.currentEdit, false, true);
      }
      if (!(error instanceof TypeError && error.message.startsWith("terminated"))) {
        throw error;
      }
    } finally {
      this.currentEdit = null;
      this.mutexAbortController = null;
    }
  }

  private async applyWorkspaceEdit(params: ApplyWorkspaceEditParams): Promise<boolean> {
    try {
      //TODO(Sma1lboy): adding client capabilities to indicate if client support this method rather than try-catch
      const result = await this.connection.sendRequest(ApplyWorkspaceEditRequest.type, params);
      return result;
    } catch (error) {
      try {
        await this.connection.workspace.applyEdit({
          edit: params.edit,
          label: params.label,
        });
        return true;
      } catch (fallbackError) {
        return false;
      }
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
    lines.push(`>>>>>>> ${stateDescription} {{markers}}[${edit.id}]`);
    markers += ">";
    // replace markers
    lines[0] = lines[0]!.replace("{{markers}}", markers);
    lines[lines.length - 1] = lines[lines.length - 1]!.replace("{{markers}}", markers);
    return lines;
  }

  private createCloseTagMatcher(tag: string): RegExp {
    let reg = `${tag}`;
    for (let length = tag.length - 1; length > 0; length--) {
      reg += "|" + tag.substring(0, length) + "$";
    }
    return new RegExp(reg, "g");
  }

  // FIXME: improve this
  public getCommentPrefix(languageId: string) {
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

  async provideNLOutlinesGenerate(params: ChatNLOutlinesParams, token: CancellationToken): Promise<boolean> {
    const document = this.documents.get(params.location.uri);
    if (!document) {
      return false;
    }
    if (!this.agent.getServerHealthState()?.chat_model) {
      throw {
        name: "ChatFeatureNotAvailableError",
        message: "Chat feature not available",
      } as ChatFeatureNotAvailableError;
    }
    const config = this.agent.getConfig().chat;
    if (isEmptyRange(params.location.range)) {
      throw {
        name: "ChatEditNoDocumentSelectedError",
        message: "No document selected",
      } as ChatEditNoDocumentSelectedError;
    }
    const documentText = document.getText();
    const selection = {
      start: document.offsetAt(params.location.range.start),
      end: document.offsetAt(params.location.range.end),
    };
    const selectedDocumentText = documentText.substring(selection.start, selection.end);
    if (selection.end - selection.start > config.edit.documentMaxChars) {
      throw { name: "ChatEditDocumentTooLongError", message: "Document too long" } as ChatEditDocumentTooLongError;
    }
    if (this.mutexAbortController) {
      throw {
        name: "ChatEditMutexError",
        message: "Another edit/NL generate is already in progress",
      } as ChatEditMutexError;
    }
    this.mutexAbortController = new AbortController();
    token.onCancellationRequested(() => this.mutexAbortController?.abort());

    const lines = selectedDocumentText.split("\n");
    const startLine = document.positionAt(selection.start).line;
    const numberedText = lines.map((line, index) => `${startLine + index + 1} | ${line}`).join("\n");
    try {
      const readableStream = await this.agent.provideNLOutlines(numberedText, {
        signal: this.mutexAbortController.signal,
      });

      if (readableStream) {
        const editId = "tabby-" + cryptoRandomString({ length: 6, type: "alphanumeric" });
        let currentLine = document.positionAt(selection.start).line;
        const endLine = document.positionAt(selection.end).line;

        let startMarker = `${this.getCommentPrefix(document.languageId)}>>> [${editId}]`;
        let endMarker = `${this.getCommentPrefix(document.languageId)}<<< [${editId}]`;

        this.currentOutlines = { uri: params.location.uri, outlines: [] };

        await this.applyWorkspaceEdit({
          edit: {
            changes: {
              [params.location.uri]: [
                {
                  range: { start: { line: currentLine, character: 0 }, end: { line: currentLine, character: 0 } },
                  newText: `${startMarker}\n`,
                },
              ],
            },
          },
        });
        currentLine++;

        let buffer = "";

        for await (const chunk of readableStream) {
          if (typeof chunk === "string") {
            buffer += chunk;

            let newlineIndex: number;
            while ((newlineIndex = buffer.indexOf("\n")) !== -1) {
              const fullLine = buffer.slice(0, newlineIndex).trim();
              buffer = buffer.slice(newlineIndex + 1);

              // Parse the line to extract line number and content
              const match = fullLine.match(/^(\d+)\s*\|\s*(.*)$/);
              if (match) {
                const [, lineNumber, content] = match;
                const parsedLineNumber = parseInt(lineNumber!, 10);

                if (!isNaN(parsedLineNumber) && content) {
                  this.currentOutlines.outlines.push({ line: parsedLineNumber - 1 + 1, text: content.trim() });

                  startMarker += ".";
                  endMarker += ".";

                  await this.applyWorkspaceEdit({
                    edit: {
                      changes: {
                        [params.location.uri]: [
                          {
                            range: {
                              start: { line: currentLine - 1, character: 0 },
                              end: { line: currentLine, character: 0 },
                            },
                            newText: `${startMarker}\n`,
                          },
                        ],
                      },
                    },
                  });

                  this.connection.sendNotification(ChatNLOutlinesSync.type, {
                    uri: params.location.uri,
                  });
                }
              }
            }
          }
        }

        // Insert final endMarker
        await this.applyWorkspaceEdit({
          edit: {
            changes: {
              [params.location.uri]: [
                {
                  range: {
                    start: { line: endLine + 1, character: 0 },
                    end: { line: endLine + 1, character: 0 },
                  },
                  newText: `${endMarker}\n`,
                },
              ],
            },
          },
        });

        if (buffer.trim()) {
          const match = buffer.trim().match(/^(\d+)\s*\|\s*(.*)$/);
          if (match) {
            const [, lineNumber, content] = match;
            const parsedLineNumber = parseInt(lineNumber!, 10);
            if (!isNaN(parsedLineNumber) && content) {
              this.currentOutlines.outlines.push({ line: parsedLineNumber - 1, text: content.trim() });
            }
          }
          this.connection.sendNotification(ChatNLOutlinesSync.type, {
            uri: params.location.uri,
          });
        }

        return true;
      }

      return false;
    } catch (error) {
      console.error("Error generating outlines:", error);
      throw error;
    } finally {
      this.mutexAbortController = null;
    }
  }

  getCurrentOutlines(): Outlines | null {
    return this.currentOutlines;
  }
}
