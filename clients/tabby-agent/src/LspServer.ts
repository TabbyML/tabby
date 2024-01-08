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
import { rootLogger } from "./logger";
import { splitLines, isCanceledError } from "./utils";

export class LspServer {
  private readonly connection = createConnection();
  private readonly documents = new TextDocuments(TextDocument);

  private readonly logger = rootLogger.child({ component: "LspServer" });

  private agent?: Agent;

  constructor() {
    this.connection.onInitialize(async (params) => {
      return await this.initialize(params);
    });
    this.connection.onShutdown(async () => {
      return await this.shutdown();
    });
    this.connection.onExit(async () => {
      return await this.exit();
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
      // TODO: use inlineCompletion prefer to completion
    }

    await this.agent.initialize({
      clientProperties: {
        session: {
          client: `${clientInfo?.name} ${clientInfo?.version ?? ""}`,
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

  async exit() {
    this.logger.debug("LSP: exit: request");
    return process.exit(0);
  }

  async showMessage(params: ShowMessageParams) {
    this.logger.debug({ params }, "LSP server notification: window/showMessage");
    this.connection.sendNotification("window/showMessage", params);
  }

  async completion(params: CompletionParams): Promise<CompletionList> {
    this.logger.debug({ params }, "LSP: textDocument/completion: request");
    if (!this.agent) {
      throw new Error(`Agent not bound.\n`);
    }

    try {
      const request = await this.buildCompletionRequest(params);
      this.logger.trace({ request }, "LSP: textDocument/completion: internalState: buildCompletionRequest");
      const response = await this.agent.provideCompletions(request);
      this.logger.trace({ response }, "LSP: textDocument/completion: internalState: provideCompletions");
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

  private async buildCompletionRequest(
    documentPosition: TextDocumentPositionParams,
    manually: boolean = false,
  ): Promise<CompletionRequest> {
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

  private async toCompletionList(
    response: CompletionResponse,
    documentPosition: TextDocumentPositionParams,
  ): Promise<CompletionList> {
    const { textDocument } = documentPosition;
    const document = this.documents.get(textDocument.uri)!;

    return {
      isIncomplete: false,
      items: response.choices.map((choice): CompletionItem => {
        const lines = splitLines(choice.text);
        const firstLine = lines[0] || "";
        const secondLine = lines[1] || "";
        return {
          label: firstLine,
          labelDetails: {
            detail: secondLine,
            description: "Tabby",
          },
          kind: CompletionItemKind.Text,
          detail: choice.text,
          sortText: " ", // let suggestion by tabby sorted to te top
          filterText: "\t", // filter text
          textEdit: {
            newText: choice.text,
            range: {
              start: document.positionAt(choice.replaceRange.start),
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
