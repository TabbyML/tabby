import type { Connection, CancellationToken, Range, URI } from "vscode-languageserver";
import { TextDocument } from "vscode-languageserver-textdocument";
import type { TextDocuments } from "../lsp/textDocuments";
import type { Feature } from "../feature";
import type { Configurations } from "../config";
import type { TabbyApiClient } from "../http/tabbyApiClient";
import {
  ChatEditToken,
  ChatEditRequest,
  ChatEditParams,
  ChatEditResolveRequest,
  ChatEditCommandRequest,
  ChatEditCommandParams,
  ChatEditCommand,
  ChatFeatureNotAvailableError,
  ChatEditDocumentTooLongError,
  ChatEditMutexError,
  ServerCapabilities,
  ChatEditResolveParams,
  ClientCapabilities,
  ReadFileParams,
  ReadFileRequest,
} from "../protocol";
import cryptoRandomString from "crypto-random-string";
import { isEmptyRange } from "../utils/range";
import { isBlank, formatPlaceholders } from "../utils/string";
import { readResponseStream, Edit, applyWorkspaceEdit, truncateFileContent } from "./utils";
import { initMutexAbortController, mutexAbortController, resetMutexAbortController } from "./global";
import { readFile } from "fs-extra";
import { getLogger } from "../logger";
import { isBrowser } from "../env";

export class ChatEditProvider implements Feature {
  private logger = getLogger("ChatEditProvider");
  private lspConnection: Connection | undefined = undefined;
  private clientCapabilities: ClientCapabilities | undefined = undefined;
  private currentEdit: Edit | undefined = undefined;

  constructor(
    private readonly configurations: Configurations,
    private readonly tabbyApiClient: TabbyApiClient,
    private readonly documents: TextDocuments<TextDocument>,
  ) {}

  initialize(connection: Connection, clientCapabilities: ClientCapabilities): ServerCapabilities {
    this.lspConnection = connection;
    this.clientCapabilities = clientCapabilities;
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

  async fetchFileContent(uri: URI, range?: Range, token?: CancellationToken) {
    this.logger.trace("Prepare to fetch text content...");
    let text: string | undefined = undefined;
    const targetDocument = this.documents.get(uri);
    if (targetDocument) {
      this.logger.trace("Fetching text content from synced text document.", {
        uri: targetDocument.uri,
        range: range,
      });
      text = targetDocument.getText(range);
      this.logger.trace("Fetched text content from synced text document.", { text });
    } else if (this.clientCapabilities?.tabby?.workspaceFileSystem) {
      const params: ReadFileParams = {
        uri: uri,
        format: "text",
        range: range
          ? {
              start: { line: range.start.line, character: 0 },
              end: { line: range.end.line, character: range.end.character },
            }
          : undefined,
      };
      this.logger.trace("Fetching text content from ReadFileRequest.", { params });
      const result = await this.lspConnection?.sendRequest(ReadFileRequest.type, params, token);
      this.logger.trace("Fetched text content from ReadFileRequest.", { result });
      text = result?.text;
    } else if (!isBrowser) {
      try {
        const content = await readFile(uri, "utf-8");
        const textDocument = TextDocument.create(uri, "text", 0, content);
        text = textDocument.getText(range);
      } catch (error) {
        this.logger.trace("Failed to fetch text content from file system.", { error });
      }
    }
    return text;
  }

  async provideEdit(params: ChatEditParams, token: CancellationToken): Promise<ChatEditToken | null> {
    if (params.format !== "previewChanges") {
      return null;
    }
    const document = this.documents.get(params.location.uri);
    if (!document) {
      return null;
    }
    if (!this.lspConnection) {
      return null;
    }
    if (!this.tabbyApiClient.isChatApiAvailable()) {
      throw {
        name: "ChatFeatureNotAvailableError",
        message: "Chat feature not available",
      } as ChatFeatureNotAvailableError;
    }
    const config = this.configurations.getMergedConfig().chat.edit;

    // FIXME(@icycodes): the command too long check is temporarily disabled,
    //    as we pass the diagnostics context as the command for now
    // if (params.command.length > config.commandMaxChars) {
    //   throw { name: "ChatEditCommandTooLongError", message: "Command too long" } as ChatEditCommandTooLongError;
    // }

    const documentText = document.getText();
    const selection = {
      start: document.offsetAt(params.location.range.start),
      end: document.offsetAt(params.location.range.end),
    };
    const selectedDocumentText = documentText.substring(selection.start, selection.end);
    if (selection.end - selection.start > config.documentMaxChars) {
      throw { name: "ChatEditDocumentTooLongError", message: "Document too long" } as ChatEditDocumentTooLongError;
    }

    if (mutexAbortController && !mutexAbortController.signal.aborted) {
      throw {
        name: "ChatEditMutexError",
        message: "Another chat edit is already in progress",
      } as ChatEditMutexError;
    }

    initMutexAbortController();
    token.onCancellationRequested(() => mutexAbortController?.abort());

    let insertMode: boolean = isEmptyRange(params.location.range);
    const presetCommand = /^\/\w+\b/g.exec(params.command)?.[0];
    if (presetCommand) {
      insertMode = config.presetCommands[presetCommand]?.kind === "insert";
    }

    let promptTemplate: string;
    let userCommand: string;
    const presetConfig = presetCommand && config.presetCommands[presetCommand];
    if (presetConfig) {
      promptTemplate = presetConfig.promptTemplate;
      userCommand = params.command.substring(presetCommand.length);
    } else {
      promptTemplate = insertMode ? config.promptTemplate.insert : config.promptTemplate.replace;
      userCommand = params.command;
    }

    // Extract the selected text and the surrounding context
    const documentSelection = documentText.substring(selection.start, selection.end);
    let documentPrefix = documentText.substring(0, selection.start);
    let documentSuffix = documentText.substring(selection.end);
    if (documentText.length > config.documentMaxChars) {
      const charsRemain = config.documentMaxChars - documentSelection.length;
      if (documentPrefix.length < charsRemain / 2) {
        documentSuffix = documentSuffix.substring(0, charsRemain - documentPrefix.length);
      } else if (documentSuffix.length < charsRemain / 2) {
        documentPrefix = documentPrefix.substring(documentPrefix.length - charsRemain + documentSuffix.length);
      } else {
        documentPrefix = documentPrefix.substring(documentPrefix.length - charsRemain / 2);
        documentSuffix = documentSuffix.substring(0, charsRemain / 2);
      }
    }

    const [fileContextListTemplate, fileContextItemTemplate] = config.fileContext.promptTemplate;
    const fileContextItems =
      (
        await Promise.all(
          (params.context ?? []).slice(0, config.fileContext.maxFiles).map(async (item) => {
            const content = await this.fetchFileContent(item.uri, item.range, token);
            if (!content || isBlank(content)) {
              return undefined;
            }
            const fileContent = truncateFileContent(content, config.fileContext.maxCharsPerFile);
            return formatPlaceholders(fileContextItemTemplate, {
              filepath: item.uri,
              referrer: item.referrer,
              content: fileContent,
            });
          }),
        )
      )
        .filter((item): item is string => item !== undefined)
        .join("\n") ?? "";

    const fileContext = !isBlank(fileContextItems)
      ? formatPlaceholders(fileContextListTemplate, {
          fileList: fileContextItems,
        })
      : "";

    const messages: { role: "user"; content: string }[] = [
      {
        role: "user",
        content: formatPlaceholders(promptTemplate, {
          filepath: params.location.uri,
          documentPrefix: documentPrefix,
          document: documentSelection,
          documentSuffix: documentSuffix,
          command: userCommand,
          languageId: document.languageId,
          fileContext: fileContext,
        }),
      },
    ];
    this.logger.debug(`messages: ${JSON.stringify(messages)}`);

    const readableStream = await this.tabbyApiClient.fetchChatStream(
      {
        messages,
        model: "",
        stream: true,
      },
      mutexAbortController?.signal,
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
    await readResponseStream(
      readableStream,
      this.lspConnection,
      this.currentEdit,
      mutexAbortController,
      () => {
        this.currentEdit = undefined;
        resetMutexAbortController();
      },
      config.responseDocumentTag,
      config.responseCommentTag,
    );
    return editId;
  }

  async stopEdit(id: ChatEditToken): Promise<void> {
    if (this.isCurrentEdit(id)) {
      mutexAbortController?.abort();
    }
  }

  async resolveEdit(params: ChatEditResolveParams): Promise<boolean> {
    if (params.action === "cancel") {
      mutexAbortController?.abort();
      return false;
    }

    const document = this.documents.get(params.location.uri);
    if (!document) {
      return false;
    }

    if (!this.lspConnection) {
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

    await applyWorkspaceEdit(
      {
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
      },
      this.lspConnection,
    );
    return true;
  }
}
