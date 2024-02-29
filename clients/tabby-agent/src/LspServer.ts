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
import { logger } from "./logger";
import { splitLines, isCanceledError } from "./utils";

export class LspServer {
  private readonly connection = createConnection();
  private readonly documents = new TextDocuments(TextDocument);

  private readonly logger = logger("LspServer");

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
    this.logger.debug({ params }, "LSP: initialize: request");
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
    return result;
  }

  async shutdown() {
    this.logger.debug("LSP: shutdown: request");
    if (!this.agent) {
      throw new Error(`Agent not bound.\n`);
    }

    await this.agent.finalize();
  }

  exit() {
    this.logger.debug("LSP: exit: request");
    return process.exit(0);
  }

  async showMessage(params: ShowMessageParams) {
    this.logger.debug({ params }, "LSP server notification: window/showMessage");
    await this.connection.sendNotification("window/showMessage", params);
  }

  async completion(params: CompletionParams): Promise<CompletionList> {
    this.logger.debug({ params }, "LSP: textDocument/completion: request");
    if (!this.agent) {
      throw new Error(`Agent not bound.\n`);
    }

    try {
      const request = this.buildCompletionRequest(params);
      const response = await this.agent.provideCompletions(request);
      const completionList = this.toCompletionList(response, params);
      this.logger.debug({ completionList }, "LSP: textDocument/completion: response");
      return completionList;
    } catch (error) {
      if (isCanceledError(error)) {
        this.logger.debug({ error }, "LSP: textDocument/completion: canceled");
      } else {
        this.logger.error({ error }, "LSP: textDocument/completion: error");
      }
    }

    return {
      isIncomplete: true,
      items: [],
    };
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
      isIncomplete: true,
      items: response.choices.map((choice): CompletionItem => {
        const insertionText = choice.text.slice(document.offsetAt(position) - choice.replaceRange.start);

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
              end: document.positionAt(choice.replaceRange.end),
            },
          },
          data: {
            completionId: response.id,
            choiceIndex: choice.index,
          },
        };
      }),
    };
  }
}
