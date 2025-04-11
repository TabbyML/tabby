import { TextDocument } from "vscode-languageserver-textdocument";
import {
  CancellationToken,
  Connection,
  Location,
  Range,
  ShowDocumentParams,
  TextDocuments,
} from "vscode-languageserver";
import type { Feature } from "../feature";
import {
  ChatEditDocumentTooLongError,
  ChatEditMutexError,
  ChatFeatureNotAvailableError,
  ServerCapabilities,
  SmartApplyRequest,
  SmartApplyParams,
} from "../protocol";
import { Configurations } from "../config";
import { TabbyApiClient } from "../http/tabbyApiClient";
import cryptoRandomString from "crypto-random-string";
import { getLogger } from "../logger";
import { readResponseStream, showDocument, Edit } from "./utils";
import { getSmartApplyRange } from "./smartRange";
import { initMutexAbortController, mutexAbortController, resetMutexAbortController } from "./global";
import { ChatFeature } from ".";

const logger = getLogger("SmartApplyFeature");

export class SmartApplyFeature implements Feature {
  private lspConnection: Connection | undefined = undefined;
  constructor(
    private readonly chat: ChatFeature,
    private readonly configurations: Configurations,
    private readonly documents: TextDocuments<TextDocument>,
  ) {}

  initialize(connection: Connection): ServerCapabilities | Promise<ServerCapabilities> {
    this.lspConnection = connection;
    connection.onRequest(SmartApplyRequest.type, async (params, token) => {
      return this.provideSmartApplyEdit(params, token);
    });

    return {};
  }
  initialized?(): void | Promise<void> {
    //nothing
  }
  shutdown?(): void | Promise<void> {
    //nothing
  }

  async provideSmartApplyEdit(params: SmartApplyParams, token: CancellationToken): Promise<boolean> {
    logger.debug("Getting document");
    const document = this.documents.get(params.location.uri);
    if (!document) {
      logger.debug("Document not found, returning false");
      return false;
    }
    if (!this.lspConnection) {
      logger.debug("LSP connection lost.");
      return false;
    }
    if (!this.chat.isAvailable()) {
      throw {
        name: "ChatFeatureNotAvailableError",
        message: "Chat feature not available",
      } as ChatFeatureNotAvailableError;
    }

    if (mutexAbortController && !mutexAbortController.signal.aborted) {
      logger.warn("Another smart edit is already in progress");
      throw {
        name: "ChatEditMutexError",
        message: "Another smart edit is already in progress",
      } as ChatEditMutexError;
    }
    initMutexAbortController();
    logger.debug("mutex abort status: " + (mutexAbortController === undefined));
    token.onCancellationRequested(() => mutexAbortController?.abort());

    let applyRange = getSmartApplyRange(document, params.text);
    //if cannot find range, lets use backend LLMs
    if (!applyRange) {
      applyRange = await provideSmartApplyLineRange(
        document,
        params.text,
        this.chat.tabbyApiClient,
        this.configurations,
      );
    }

    if (!applyRange) {
      return false;
    }

    try {
      //reveal editor range
      const revealEditorRangeParams: ShowDocumentParams = {
        uri: params.location.uri,
        selection: {
          start: applyRange.range.start,
          end: applyRange.range.end,
        },
        takeFocus: true,
      };
      await showDocument(revealEditorRangeParams, this.lspConnection);
    } catch (error) {
      logger.warn("cline not support reveal range");
    }

    try {
      await provideSmartApplyEditLLM(
        {
          uri: params.location.uri,
          range: {
            start: applyRange.range.start,
            end: { line: applyRange.range.end.line + 1, character: 0 },
          },
        },
        params.text,
        applyRange.action === "insert" ? true : false,
        document,
        this.lspConnection,
        this.chat.tabbyApiClient,
        this.configurations,
        mutexAbortController,
        () => {
          resetMutexAbortController();
        },
      );
      return true;
    } catch (error) {
      logger.error("Error applying smart edit:", error);
      return false;
    } finally {
      logger.debug("Resetting mutex abort controller");
      resetMutexAbortController();
    }
  }
}

async function provideSmartApplyLineRange(
  document: TextDocument,
  applyCodeBlock: string,
  tabbyApiClient: TabbyApiClient,
  configurations: Configurations,
): Promise<{ range: Range; action: "insert" | "replace" } | undefined> {
  if (!document) {
    return undefined;
  }

  const documentText = document
    .getText()
    .split("\n")
    .map((line, idx) => `${idx + 1} | ${line}`)
    .join("\n");

  const config = configurations.getMergedConfig();
  const promptTemplate = config.chat.smartApplyLineRange.promptTemplate;

  const messages: { role: "user"; content: string }[] = [
    {
      role: "user",
      content: promptTemplate.replace(/{{document}}|{{applyCode}}/g, (pattern: string) => {
        switch (pattern) {
          case "{{document}}":
            return documentText;
          case "{{applyCode}}":
            return applyCodeBlock;
          default:
            return "";
        }
      }),
    },
  ];

  try {
    const readableStream = await tabbyApiClient.fetchChatStream({
      messages,
      model: "",
      stream: true,
    });

    if (!readableStream) {
      return undefined;
    }

    let response = "";
    for await (const chunk of readableStream) {
      response += chunk;
    }

    const regex = /<GENERATEDCODE>(.*?)<\/GENERATEDCODE>/s;
    const match = response.match(regex);
    if (match && match[1]) {
      response = match[1].trim();
    }

    const range = response.split("-");
    if (range.length !== 2) {
      return undefined;
    }

    const startLine = parseInt(range[0] ?? "0", 10) - 1;
    const endLine = parseInt(range[1] ?? "0", 10) - 1;

    return {
      range: {
        start: { line: startLine < 0 ? 0 : startLine, character: 0 },
        end: { line: endLine < 0 ? 0 : endLine, character: Number.MAX_SAFE_INTEGER },
      },
      action: startLine == endLine ? "insert" : "replace",
    };
  } catch (error) {
    return undefined;
  }
}

async function provideSmartApplyEditLLM(
  location: Location,
  applyCode: string,
  insertMode: boolean,
  document: TextDocument,
  lspConnection: Connection,
  tabbyApiClient: TabbyApiClient,
  configurations: Configurations,
  mutexAbortController: AbortController | undefined,
  onResetMutex: () => void,
): Promise<boolean> {
  if (!document) {
    logger.warn("Document not found");
    return false;
  }
  if (!lspConnection) {
    logger.warn("LSP connection failed");
    return false;
  }

  const config = configurations.getMergedConfig();
  const documentText = document.getText();
  const selection = {
    start: document.offsetAt(location.range.start),
    end: document.offsetAt(location.range.end),
  };
  const selectedDocumentText = documentText.substring(selection.start, selection.end);

  logger.debug("current selectedDoc: " + selectedDocumentText);

  if (selection.end - selection.start > config.chat.edit.documentMaxChars) {
    throw { name: "ChatEditDocumentTooLongError", message: "Document too long" } as ChatEditDocumentTooLongError;
  }

  const promptTemplate = config.chat.smartApply.promptTemplate;

  // Extract the selected text and the surrounding context
  let documentPrefix = documentText.substring(0, selection.start);
  let documentSuffix = documentText.substring(selection.end);
  if (documentText.length > config.chat.edit.documentMaxChars) {
    const charsRemain = config.chat.edit.documentMaxChars - selectedDocumentText.length;
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
      content: promptTemplate.replace(/{{document}}|{{code}}/g, (pattern: string) => {
        switch (pattern) {
          case "{{document}}":
            return selectedDocumentText;
          case "{{code}}":
            return applyCode || "";
          default:
            return "";
        }
      }),
    },
  ];

  try {
    const readableStream = await tabbyApiClient.fetchChatStream({
      messages,
      model: "",
      stream: true,
    });

    if (!readableStream) {
      return false;
    }
    const editId = "tabby-" + cryptoRandomString({ length: 6, type: "alphanumeric" });
    const currentEdit: Edit = {
      id: editId,
      location: location,
      languageId: document.languageId,
      originalText: selectedDocumentText,
      editedRange: insertMode
        ? { start: location.range.start, end: location.range.end }
        : { start: location.range.start, end: location.range.end },
      editedText: "",
      comments: "",
      buffer: "",
      state: "editing",
    };

    await readResponseStream(
      readableStream,
      lspConnection,
      currentEdit,
      mutexAbortController,
      onResetMutex,
      config.chat.edit.responseDocumentTag,
      config.chat.edit.responseCommentTag,
    );

    return true;
  } catch (error) {
    return false;
  }
}
