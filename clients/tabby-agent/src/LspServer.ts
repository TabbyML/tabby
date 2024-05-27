import {
  createConnection,
  TextDocuments,
  TextDocumentSyncKind,
  InitializeParams,
  InitializeResult,
  ShowMessageParams,
  MessageType,
  CompletionParams,
  CompletionList,
  CompletionItem,
  CompletionItemKind,
  TextDocumentPositionParams,
} from "vscode-languageserver/node";
import { TextDocument } from "vscode-languageserver-textdocument";
import { name as agentName, version as agentVersion } from "../package.json";
import { Agent, StatusChangedEvent, CompletionRequest, CompletionResponse } from "./Agent";
import { getLogger } from "./logger";
import { splitLines, isCanceledError } from "./utils";

export class LspServer {
  private readonly connection = createConnection();
  private readonly documents = new TextDocuments(TextDocument);

  private readonly logger = getLogger("LSP");

  private agent?: Agent;

  constructor() {
    this.connection.onInitialize(async (params) => {
      return await this.initialize(params);
    });
    this.connection.onShutdown(async () => {
      return await this.shutdown();
    });
    this.connection.onExit(async () => {
      return this.exit();
    });
    this.connection.onCompletion(async (params) => {
      return await this.completion(params);
    });
  }

  bind(agent: Agent): void {
    this.agent = agent;

    this.agent.on("statusChanged", (event: StatusChangedEvent) => {
      if (event.status === "disconnected" || event.status === "unauthorized") {
        this.showMessage({
          type: MessageType.Warning,
          message: `Tabby agent status: ${event.status}`,
        });
      }
    });
  }

  listen() {
    this.documents.listen(this.connection);
    this.connection.listen();
  }

  // LSP interface methods

  async initialize(params: InitializeParams): Promise<InitializeResult> {
    this.logger.debug("[-->] Initialize Request");
    this.logger.trace("Initialize params:", params);
    if (!this.agent) {
      throw new Error(`Agent not bound.\n`);
    }

    const { clientInfo, capabilities } = params;
    if (capabilities.textDocument?.inlineCompletion) {
      // TODO: use inlineCompletion instead of completion
    }

    await this.agent.initialize({
      clientProperties: {
        session: {
          client: `${clientInfo?.name} ${clientInfo?.version ?? ""}`,
          ide: {
            name: clientInfo?.name,
            version: clientInfo?.version,
          },
          tabby_plugin: {
            name: `${agentName} (LSP)`,
            version: agentVersion,
          },
        },
      },
    });

    const result: InitializeResult = {
      capabilities: {
        textDocumentSync: {
          openClose: true,
          change: TextDocumentSyncKind.Incremental,
        },
        completionProvider: {},
        // inlineCompletionProvider: {},
      },
      serverInfo: {
        name: agentName,
        version: agentVersion,
      },
    };
    this.logger.debug("[<--] Initialize Response");
    this.logger.trace("Initialize result:", result);
    return result;
  }

  async shutdown() {
    this.logger.debug("[-->] shutdown");
    if (!this.agent) {
      throw new Error(`Agent not bound.\n`);
    }

    await this.agent.finalize();
    this.logger.debug("[<--] shutdown");
  }

  exit() {
    this.logger.debug("[-->] exit");
    return process.exit(0);
  }

  async showMessage(params: ShowMessageParams) {
    this.logger.debug("[<--] window/showMessage");
    this.logger.trace("ShowMessage params:", params);
    await this.connection.sendNotification("window/showMessage", params);
  }

  async completion(params: CompletionParams): Promise<CompletionList> {
    this.logger.debug("[-->] textDocument/completion");
    this.logger.trace("Completion params:", params);
    if (!this.agent) {
      throw new Error(`Agent not bound.\n`);
    }

    let completionList: CompletionList = {
      isIncomplete: true,
      items: [],
    };
    try {
      const request = this.buildCompletionRequest(params);
      const response = await this.agent.provideCompletions(request);
      completionList = this.toCompletionList(response, params);
    } catch (error) {
      if (isCanceledError(error)) {
        this.logger.debug("Completion request canceled.");
      } else {
        this.logger.error("Completion request failed.", error);
      }
    }

    this.logger.debug("[<--] textDocument/completion");
    this.logger.trace("Completion result:", completionList);
    return completionList;
  }

  private buildCompletionRequest(
    documentPosition: TextDocumentPositionParams,
    manually: boolean = false,
  ): CompletionRequest {
    const { textDocument, position } = documentPosition;
    const document = this.documents.get(textDocument.uri)!;

    const request: CompletionRequest = {
      filepath: document.uri,
      language: document.languageId,
      text: document.getText(),
      position: document.offsetAt(position),
      manually,
    };
    return request;
  }

  private toCompletionList(response: CompletionResponse, documentPosition: TextDocumentPositionParams): CompletionList {
    const { textDocument, position } = documentPosition;
    const document = this.documents.get(textDocument.uri)!;

    // Get word prefix if cursor is at end of a word
    const linePrefix = document.getText({
      start: { line: position.line, character: 0 },
      end: position,
    });
    const wordPrefix = linePrefix.match(/(\w+)$/)?.[0] ?? "";

    return {
      isIncomplete: response.isIncomplete,
      items: response.items.map((item): CompletionItem => {
        const insertionText = item.insertText.slice(document.offsetAt(position) - item.range.start);

        const lines = splitLines(insertionText);
        const firstLine = lines[0] || "";
        const secondLine = lines[1] || "";
        return {
          label: wordPrefix + firstLine,
          labelDetails: {
            detail: secondLine,
            description: "Tabby",
          },
          kind: CompletionItemKind.Text,
          documentation: {
            kind: "markdown",
            value: `\`\`\`\n${linePrefix + insertionText}\n\`\`\`\n ---\nSuggested by Tabby.`,
          },
          textEdit: {
            newText: wordPrefix + insertionText,
            range: {
              start: { line: position.line, character: position.character - wordPrefix.length },
              end: document.positionAt(item.range.end),
            },
          },
          data: item.data,
        };
      }),
    };
  }
}
