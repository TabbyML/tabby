import type { Range, Location, Connection, CancellationToken, WorkspaceEdit } from "vscode-languageserver";
import type { TextDocument } from "vscode-languageserver-textdocument";
import type { TextDocuments } from "../lsp/textDocuments";
import type { Feature } from "../feature";
import type { Configurations } from "../config";
import type { TabbyApiClient } from "../http/tabbyApiClient";
import type { Readable } from "readable-stream";
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
  ChatEditMutexError,
  ApplyWorkspaceEditRequest,
  ApplyWorkspaceEditParams,
  ServerCapabilities,
} from "../protocol";
import cryptoRandomString from "crypto-random-string";
import * as Diff from "diff";
import { isEmptyRange } from "../utils/range";
import { isBlank } from "../utils/string";

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

export class ChatEditProvider implements Feature {
  private lspConnection: Connection | undefined = undefined;
  private currentEdit: Edit | undefined = undefined;
  private mutexAbortController: AbortController | undefined = undefined;

  constructor(
    private readonly configurations: Configurations,
    private readonly tabbyApiClient: TabbyApiClient,
    private readonly documents: TextDocuments<TextDocument>,
  ) {}

  initialize(connection: Connection): ServerCapabilities {
    this.lspConnection = connection;
    connection.onRequest(ChatEditCommandRequest.type, async (params) => {
      return this.provideEditCommands(params);
    });
    connection.onRequest(ChatEditRequest.type, async (params, token) => {
      return this.provideEdit(params, token);
    });
    connection.onRequest(ChatEditResolveRequest.type, async (params) => {
      return this.resolveEdit(params);
    });
    return {};
  }

  private isCurrentEdit(id: ChatEditToken): boolean {
    return this.currentEdit?.id === id;
  }

  async provideEditCommands(params: ChatEditCommandParams): Promise<ChatEditCommand[]> {
    const config = this.configurations.getMergedConfig();
    const commands = config.chat.edit.presetCommands;
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
    if (!this.tabbyApiClient.isChatApiAvailable()) {
      throw {
        name: "ChatFeatureNotAvailableError",
        message: "Chat feature not available",
      } as ChatFeatureNotAvailableError;
    }
    const config = this.configurations.getMergedConfig();

    // FIXME(@icycodes): the command too long check is temporarily disabled,
    //    as we pass the diagnostics context as the command for now
    // if (params.command.length > config.chat.edit.commandMaxChars) {
    //   throw { name: "ChatEditCommandTooLongError", message: "Command too long" } as ChatEditCommandTooLongError;
    // }

    const documentText = document.getText();
    const selection = {
      start: document.offsetAt(params.location.range.start),
      end: document.offsetAt(params.location.range.end),
    };
    const selectedDocumentText = documentText.substring(selection.start, selection.end);
    if (selection.end - selection.start > config.chat.edit.documentMaxChars) {
      throw { name: "ChatEditDocumentTooLongError", message: "Document too long" } as ChatEditDocumentTooLongError;
    }

    if (this.mutexAbortController && !this.mutexAbortController.signal.aborted) {
      throw {
        name: "ChatEditMutexError",
        message: "Another smart edit is already in progress",
      } as ChatEditMutexError;
    }

    this.mutexAbortController = new AbortController();
    token.onCancellationRequested(() => this.mutexAbortController?.abort());

    let insertMode: boolean = isEmptyRange(params.location.range);
    const presetCommand = /^\/\w+\b/g.exec(params.command)?.[0];
    if (presetCommand) {
      insertMode = config.chat.edit.presetCommands[presetCommand]?.kind === "insert";
    }

    let promptTemplate: string;
    let userCommand: string;
    const presetConfig = presetCommand && config.chat.edit.presetCommands[presetCommand];
    if (presetConfig) {
      promptTemplate = presetConfig.promptTemplate;
      userCommand = params.command.substring(presetCommand.length);
    } else {
      promptTemplate = insertMode ? config.chat.edit.promptTemplate.insert : config.chat.edit.promptTemplate.replace;
      userCommand = params.command;
    }

    // Extract the selected text and the surrounding context
    const documentSelection = documentText.substring(selection.start, selection.end);
    let documentPrefix = documentText.substring(0, selection.start);
    let documentSuffix = documentText.substring(selection.end);
    if (documentText.length > config.chat.edit.documentMaxChars) {
      const charsRemain = config.chat.edit.documentMaxChars - documentSelection.length;
      if (documentPrefix.length < charsRemain / 2) {
        documentSuffix = documentSuffix.substring(0, charsRemain - documentPrefix.length);
      } else if (documentSuffix.length < charsRemain / 2) {
        documentPrefix = documentPrefix.substring(documentPrefix.length - charsRemain + documentSuffix.length);
      } else {
        documentPrefix = documentPrefix.substring(documentPrefix.length - charsRemain / 2);
        documentSuffix = documentSuffix.substring(0, charsRemain / 2);
      }
    }

    const messages: { role: "user"; content: string }[] = [
      {
        role: "user",
        content: promptTemplate.replace(
          /{{filepath}}|{{documentPrefix}}|{{document}}|{{documentSuffix}}|{{command}}|{{languageId}}/g,
          (pattern: string) => {
            switch (pattern) {
              case "{{filepath}}":
                return params.location.uri;
              case "{{documentPrefix}}":
                return documentPrefix;
              case "{{document}}":
                return documentSelection;
              case "{{documentSuffix}}":
                return documentSuffix;
              case "{{command}}":
                return userCommand;
              case "{{languageId}}":
                return document.languageId;
              default:
                return "";
            }
          },
        ),
      },
    ];
    const readableStream = await this.tabbyApiClient.fetchChatStream(
      {
        messages,
        model: "",
        stream: true,
      },
      this.mutexAbortController.signal,
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
    await this.readResponseStream(
      readableStream,
      config.chat.edit.responseDocumentTag,
      config.chat.edit.responseCommentTag,
    );
    return editId;
  }

  async stopEdit(id: ChatEditToken): Promise<void> {
    if (this.isCurrentEdit(id)) {
      this.mutexAbortController?.abort();
    }
  }

  async resolveEdit(params: ChatEditResolveParams): Promise<boolean> {
    if (params.action === "cancel") {
      this.mutexAbortController?.abort();
      return false;
    }

    const document = this.documents.get(params.location.uri);
    if (!document) {
      return false;
    }

    let markers;
    let line = params.location.range.start.line;
    for (; line < document.lineCount; line++) {
      const lineText = document.getText({
        start: { line, character: 0 },
        end: { line: line + 1, character: 0 },
      });

      const match = /^>>>>>>> (tabby-[0-9|a-z|A-Z]{6}) (\[.*\])/g.exec(lineText);
      markers = match?.[2];
      if (markers) {
        break;
      }
    }

    if (!markers) {
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

        await this.applyWorkspaceEdit({
          edit: workspaceEdit,
          options: {
            undoStopBefore: true,
            undoStopAfter: false,
          },
        });

        edit.editedRange = {
          start: { line: edit.editedRange.start.line + 1, character: 0 },
          end: { line: edit.editedRange.end.line + 1, character: 0 },
        };
      }

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
          undoStopBefore: false,
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
      if (!this.currentEdit) {
        throw new Error("No current edit");
      }

      let inTag: "document" | "comment" | false = false;

      // Insert the first line as early as possible so codelens can be shown
      await applyEdit(this.currentEdit, true, false);

      for await (const item of stream) {
        if (!this.mutexAbortController || this.mutexAbortController.signal.aborted) {
          break;
        }
        const delta = typeof item === "string" ? item : "";
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
          if (delta.includes("\n")) {
            await applyEdit(edit, false, false);
          }
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
      this.currentEdit = undefined;
      this.mutexAbortController = undefined;
    }
  }

  private async applyWorkspaceEdit(params: ApplyWorkspaceEditParams): Promise<boolean> {
    const lspConnection = this.lspConnection;
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
  // [x] stopped
  // footer line
  // >>>>>>> End of changes
  private generateChangesPreview(edit: Edit): string[] {
    const lines: string[] = [];
    let markers = "";
    // lines.push(`<<<<<<< ${stateDescription} {{markers}}[${edit.id}]`);
    markers += "[";
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
        if (edit.state === "stopped") {
          pushDiffValue(lastDiff.value, "x");
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
          pushDiffValue(diff.value, "x");
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

  private createCloseTagMatcher(tag: string): RegExp {
    let reg = `${tag}`;
    for (let length = tag.length - 1; length > 0; length--) {
      reg += "|" + tag.substring(0, length) + "$";
    }
    return new RegExp(reg, "g");
  }

  // FIXME: improve this
  private getCommentPrefix(languageId: string) {
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
}
