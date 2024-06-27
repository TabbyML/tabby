import { createConnection as nodeCreateConnection } from "vscode-languageserver/node";
import {
  createConnection as browserCreateConnection,
  BrowserMessageReader,
  BrowserMessageWriter,
} from "vscode-languageserver/browser";
import {
  Position,
  Range,
  Location,
  ProposedFeatures,
  CancellationToken,
  RegistrationRequest,
  UnregistrationRequest,
  TextDocumentSyncKind,
  TextDocumentPositionParams,
  TextDocumentContentChangeEvent,
  NotebookDocument,
  NotebookDocuments,
  NotebookCell,
  CompletionParams,
  CompletionTriggerKind,
  CompletionItemKind,
  InlineCompletionParams,
  InlineCompletionTriggerKind,
} from "vscode-languageserver";
import {
  InitializeParams,
  InitializeResult,
  ClientInfo,
  ClientCapabilities,
  ServerCapabilities,
  DidChangeConfigurationParams,
  ClientProvidedConfig,
  AgentServerInfoRequest,
  AgentServerInfoSync,
  ServerInfo,
  AgentStatusRequest,
  AgentStatusSync,
  Status,
  AgentIssuesRequest,
  AgentIssuesSync,
  IssueList,
  AgentIssueDetailRequest,
  IssueDetailParams,
  IssueDetailResult,
  CompletionList,
  CompletionItem,
  InlineCompletionRequest,
  InlineCompletionList,
  InlineCompletionItem,
  ChatFeatureRegistration,
  GenerateCommitMessageRequest,
  GenerateCommitMessageParams,
  GenerateCommitMessageResult,
  TelemetryEventNotification,
  EventParams,
  ChatFeatureNotAvailableError,
  EditorOptions,
  GitRepositoryRequest,
  GitRepositoryParams,
  GitRepository,
  GitDiffRequest,
  GitDiffParams,
  GitDiffResult,
  DataStoreGetRequest,
  DataStoreGetParams,
  DataStoreSetRequest,
  DataStoreSetParams,
  EditorOptionsRequest,
  ReadFileRequest,
  ReadFileParams,
  LanguageSupportDeclarationRequest,
  LanguageSupportSemanticTokensRangeRequest,
} from "./protocol";
import { TextDocuments } from "./TextDocuments";
import { TextDocument } from "vscode-languageserver-textdocument";
import deepEqual from "deep-equal";
import type {
  AgentIssue,
  ConfigUpdatedEvent,
  StatusChangedEvent,
  IssuesUpdatedEvent,
  CompletionRequest,
  CompletionResponse,
  ClientProperties,
} from "../Agent";
import { TabbyAgent } from "../TabbyAgent";
import type { PartialAgentConfig } from "../AgentConfig";
import { isBrowser } from "../env";
import { getLogger, Logger } from "../logger";
import type { DataStore } from "../dataStore";
import { name as agentName, version as agentVersion } from "../../package.json";
import { RecentlyChangedCodeSearch } from "../codeSearch/RecentlyChangedCodeSearch";
import { isPositionInRange, intersectionRange } from "../utils/range";
import { extractNonReservedWordList } from "../utils/string";
import { splitLines, isBlank } from "../utils";
import { ChatEditProvider } from "./ChatEditProvider";
import { CodeLensProvider } from "./CodeLensProvider";
import { CommandProvider } from "./CommandProvider";

export class Server {
  private readonly logger = getLogger("LspServer");
  private readonly connection = isBrowser
    ? browserCreateConnection(ProposedFeatures.all, new BrowserMessageReader(self), new BrowserMessageWriter(self))
    : nodeCreateConnection(ProposedFeatures.all);

  private readonly documents = new TextDocuments(TextDocument);
  private readonly notebooks = new NotebookDocuments(this.documents);
  private recentlyChangedCodeSearch: RecentlyChangedCodeSearch | undefined = undefined;
  private chatEditProvider: ChatEditProvider;
  private codeLensProvider: CodeLensProvider | undefined = undefined;
  private commandProvider: CommandProvider;

  private clientInfo?: ClientInfo | undefined | null;
  private clientCapabilities?: ClientCapabilities | undefined | null;
  private clientProvidedConfig?: ClientProvidedConfig | undefined | null;
  private serverInfo?: ServerInfo | undefined | null;

  constructor(private readonly agent: TabbyAgent) {
    // Lifecycle
    this.connection.onInitialize(async (params) => {
      return this.initialize(params);
    });
    this.connection.onInitialized(async () => {
      return this.initialized();
    });
    this.connection.onDidChangeConfiguration(async (params) => {
      return this.updateConfiguration(params);
    });
    this.connection.onShutdown(async () => {
      return this.shutdown();
    });
    this.connection.onExit(async () => {
      return this.exit();
    });
    // Agent
    this.connection.onRequest(AgentServerInfoRequest.type, async () => {
      return this.getServerInfo();
    });
    this.connection.onRequest(AgentStatusRequest.type, async () => {
      return this.getStatus();
    });
    this.connection.onRequest(AgentIssuesRequest.type, async () => {
      return this.getIssues();
    });
    this.connection.onRequest(AgentIssueDetailRequest.type, async (params) => {
      return this.getIssueDetail(params);
    });
    // Documents Sync
    this.documents.listen(this.connection);
    this.notebooks.listen(this.connection);

    // Completion
    this.connection.onCompletion(async (params, token) => {
      return this.provideCompletion(params, token);
    });
    this.connection.onRequest(InlineCompletionRequest.type, async (params, token) => {
      return this.provideInlineCompletion(params, token);
    });
    // Chat
    this.chatEditProvider = new ChatEditProvider(this.connection, this.documents, this.agent);
    this.connection.onRequest(GenerateCommitMessageRequest.type, async (params, token) => {
      return this.generateCommitMessage(params, token);
    });
    // Telemetry
    this.connection.onNotification(TelemetryEventNotification.type, async (param) => {
      return this.event(param);
    });
    // Command
    this.commandProvider = new CommandProvider(this.connection, this.chatEditProvider);
  }

  listen() {
    this.connection.listen();
  }

  private async initialize(params: InitializeParams): Promise<InitializeResult> {
    const clientInfo: ClientInfo | undefined = params.clientInfo;
    this.clientInfo = clientInfo;
    const clientCapabilities: ClientCapabilities = params.capabilities;
    this.clientCapabilities = clientCapabilities;
    const clientProvidedConfig: ClientProvidedConfig | undefined = params.initializationOptions?.config;
    this.clientProvidedConfig = clientProvidedConfig;
    const serverCapabilities: ServerCapabilities = {
      textDocumentSync: {
        openClose: true,
        change: TextDocumentSyncKind.Incremental,
      },
      notebookDocumentSync: {
        notebookSelector: [
          {
            notebook: "*",
          },
        ],
      },
      workspace: {
        workspaceFolders: {
          supported: true,
          changeNotifications: true,
        },
      },
      completionProvider: undefined,
      inlineCompletionProvider: undefined,
      tabby: {
        chat: false,
      },
    };

    if (clientCapabilities.textDocument?.inlineCompletion) {
      serverCapabilities.inlineCompletionProvider = true;
    } else {
      serverCapabilities.completionProvider = {};
    }

    if (clientCapabilities.workspace?.codeLens) {
      this.codeLensProvider = new CodeLensProvider(this.connection, this.documents);
      this.codeLensProvider.fillServerCapabilities(serverCapabilities);
    }
    this.commandProvider.fillServerCapabilities(serverCapabilities);

    await this.agent.initialize({
      config: this.createInitConfig(clientProvidedConfig),
      clientProperties: this.createInitClientProperties(clientInfo, clientProvidedConfig),
      dataStore: clientCapabilities.tabby?.dataStore ? this.createDataStore() : undefined,
      loggers: [this.createLogger()],
    });

    if (this.agent.getServerHealthState()?.chat_model) {
      serverCapabilities.tabby = {
        ...serverCapabilities.tabby,
        chat: true,
      };
    }

    const result: InitializeResult = {
      capabilities: serverCapabilities,
      serverInfo: {
        name: agentName,
        version: agentVersion,
      },
    };
    return result;
  }

  private async initialized(): Promise<void> {
    const agentConfig = this.agent.getConfig();
    if (agentConfig.completion.prompt.collectSnippetsFromRecentChangedFiles.enabled) {
      this.recentlyChangedCodeSearch = new RecentlyChangedCodeSearch(
        agentConfig.completion.prompt.collectSnippetsFromRecentChangedFiles.indexing,
      );
      this.documents.onDidChangeContent(async (params: unknown) => {
        if (!params || typeof params !== "object" || !("document" in params) || !("contentChanges" in params)) {
          return;
        }
        const event = params as { document: TextDocument; contentChanges: TextDocumentContentChangeEvent[] };
        this.recentlyChangedCodeSearch?.handleDidChangeTextDocument(event);
      });
    }

    this.serverInfo = {
      config: agentConfig.server,
      health: this.agent.getServerHealthState(),
    };
    this.agent.on("configUpdated", (event: ConfigUpdatedEvent) => {
      const serverInfo = {
        config: event.config.server,
        health: this.agent.getServerHealthState(),
      };
      if (!deepEqual(serverInfo, this.serverInfo)) {
        if (this.clientCapabilities?.tabby?.agent) {
          this.connection.sendNotification(AgentServerInfoSync.type, { serverInfo });
        }
        this.serverInfo = serverInfo;
      }
    });

    this.agent.on("statusChanged", (event: StatusChangedEvent) => {
      if (this.clientCapabilities?.tabby?.agent) {
        this.connection.sendNotification(AgentStatusSync.type, {
          status: event.status,
        });
      }

      const health = this.agent.getServerHealthState();
      const serverInfo = {
        config: this.agent.getConfig().server,
        health,
      };
      if (!deepEqual(serverInfo, this.serverInfo)) {
        if (this.clientCapabilities?.tabby?.agent) {
          this.connection.sendNotification(AgentServerInfoSync.type, { serverInfo });
        }
        this.serverInfo = serverInfo;
      }
      if (health?.chat_model) {
        this.connection.sendRequest(RegistrationRequest.type, {
          registrations: [
            {
              id: ChatFeatureRegistration.type.method,
              method: ChatFeatureRegistration.type.method,
            },
          ],
        });
      } else {
        this.connection.sendRequest(UnregistrationRequest.type, {
          unregisterations: [
            {
              id: ChatFeatureRegistration.type.method,
              method: ChatFeatureRegistration.type.method,
            },
          ],
        });
      }
    });

    if (this.clientCapabilities?.tabby?.agent) {
      this.agent.on("issuesUpdated", (event: IssuesUpdatedEvent) => {
        this.connection.sendNotification(AgentIssuesSync.type, {
          issues: event.issues,
        });
      });
    }
  }

  private async updateConfiguration(params: DidChangeConfigurationParams) {
    const clientProvidedConfig: ClientProvidedConfig | null = params.settings;
    if (
      clientProvidedConfig?.server?.endpoint !== undefined &&
      clientProvidedConfig.server.endpoint !== this.clientProvidedConfig?.server?.endpoint
    ) {
      if (clientProvidedConfig.server.endpoint.trim().length > 0) {
        this.agent.updateConfig("server.endpoint", clientProvidedConfig.server.endpoint);
      } else {
        this.agent.clearConfig("server.endpoint");
      }
    }
    if (
      clientProvidedConfig?.server?.token !== undefined &&
      clientProvidedConfig.server.token !== this.clientProvidedConfig?.server?.token
    ) {
      if (clientProvidedConfig.server.token.trim().length > 0) {
        this.agent.updateConfig("server.token", clientProvidedConfig.server.token);
      } else {
        this.agent.clearConfig("server.token");
      }
    }
    if (
      clientProvidedConfig?.anonymousUsageTracking?.disable !== undefined &&
      clientProvidedConfig.anonymousUsageTracking.disable !== this.clientProvidedConfig?.anonymousUsageTracking?.disable
    ) {
      if (clientProvidedConfig.anonymousUsageTracking.disable) {
        this.agent.updateConfig("anonymousUsageTracking.disable", true);
      } else {
        this.agent.clearConfig("anonymousUsageTracking.disable");
      }
    }
    const clientType = this.getClientType(this.clientInfo);
    if (
      clientProvidedConfig?.inlineCompletion?.triggerMode !== undefined &&
      clientProvidedConfig.inlineCompletion.triggerMode !== this.clientProvidedConfig?.inlineCompletion?.triggerMode
    ) {
      this.agent.updateClientProperties(
        "user",
        `${clientType}.triggerMode`,
        clientProvidedConfig.inlineCompletion?.triggerMode,
      );
    }
    if (
      clientProvidedConfig?.keybindings !== undefined &&
      clientProvidedConfig.keybindings !== this.clientProvidedConfig?.keybindings
    ) {
      this.agent.updateClientProperties("user", `${clientType}.keybindings`, clientProvidedConfig.keybindings);
    }
    this.clientProvidedConfig = clientProvidedConfig;
  }

  private async shutdown() {
    await this.agent.finalize();
  }

  private exit() {
    return process.exit(0);
  }

  private async getServerInfo(): Promise<ServerInfo> {
    return (
      this.serverInfo ?? {
        config: this.agent.getConfig().server,
        health: this.agent.getServerHealthState(),
      }
    );
  }

  private async getStatus(): Promise<Status> {
    return this.agent.getStatus();
  }

  private async getIssues(): Promise<IssueList> {
    return { issues: this.agent.getIssues() };
  }

  private async getIssueDetail(params: IssueDetailParams): Promise<IssueDetailResult | null> {
    const detail = this.agent.getIssueDetail({ name: params.name });
    if (!detail) {
      return null;
    }
    return {
      name: detail.name,
      helpMessage: this.buildHelpMessage(detail, params.helpMessageFormat),
    };
  }

  private async provideCompletion(params: CompletionParams, token: CancellationToken): Promise<CompletionList | null> {
    if (token.isCancellationRequested) {
      return null;
    }
    const abortController = new AbortController();
    token.onCancellationRequested(() => abortController.abort());
    try {
      const result = await this.completionParamsToCompletionRequest(params, token);
      if (!result) {
        return null;
      }
      const response = await this.agent.provideCompletions(result.request, { signal: abortController.signal });
      return this.toCompletionList(response, params, result.additionalPrefixLength);
    } catch (error) {
      return null;
    }
  }

  private async provideInlineCompletion(
    params: InlineCompletionParams,
    token: CancellationToken,
  ): Promise<InlineCompletionList | null> {
    if (token.isCancellationRequested) {
      return null;
    }
    const abortController = new AbortController();
    token.onCancellationRequested(() => abortController.abort());
    try {
      const result = await this.inlineCompletionParamsToCompletionRequest(params, token);
      if (!result) {
        return null;
      }
      const response = await this.agent.provideCompletions(result.request, { signal: abortController.signal });
      return this.toInlineCompletionList(response, params, result.additionalPrefixLength);
    } catch (error) {
      return null;
    }
  }

  private async generateCommitMessage(
    params: GenerateCommitMessageParams,
    token: CancellationToken,
  ): Promise<GenerateCommitMessageResult | null> {
    if (token.isCancellationRequested) {
      return null;
    }

    if (!this.agent.getServerHealthState()?.chat_model) {
      throw { name: "ChatFeatureNotAvailableError" } as ChatFeatureNotAvailableError;
    }
    const abortController = new AbortController();
    token.onCancellationRequested(() => abortController.abort());
    const { repository } = params;
    let diffResult: GitDiffResult | undefined | null = undefined;
    if (this.clientCapabilities?.tabby?.gitProvider) {
      const params: GitDiffParams = { repository, cached: true };
      diffResult = await this.connection.sendRequest(GitDiffRequest.type, params);
      if (
        !diffResult?.diff ||
        (typeof diffResult.diff === "string" && isBlank(diffResult.diff)) ||
        (Array.isArray(diffResult.diff) && isBlank(diffResult.diff.join("")))
      ) {
        // Use uncached diff if cached diff is empty
        const params: GitDiffParams = { repository, cached: false };
        diffResult = await this.connection.sendRequest(GitDiffRequest.type, params);
      }
    } else {
      //FIXME: fallback to system `git` command
    }
    if (!diffResult || !diffResult.diff) {
      return null;
    }
    try {
      const commitMessage = await this.agent.generateCommitMessage(diffResult.diff, { signal: abortController.signal });
      return { commitMessage };
    } catch (error) {
      return null;
    }
  }

  private async event(params: EventParams): Promise<void> {
    try {
      const request = {
        type: params.type,
        select_kind: params.selectKind,
        completion_id: params.eventId.completionId,
        choice_index: params.eventId.choiceIndex,
        view_id: params.viewId,
        elapsed: params.elapsed,
      };
      await this.agent.postEvent(request);
    } catch (error) {
      return;
    }
  }

  private createInitConfig(clientProvidedConfig: ClientProvidedConfig | undefined): PartialAgentConfig {
    const config: PartialAgentConfig = {};
    if (clientProvidedConfig?.server?.endpoint && clientProvidedConfig.server.endpoint.trim().length > 0) {
      config.server = {
        endpoint: clientProvidedConfig.server.endpoint,
      };
    }
    if (clientProvidedConfig?.server?.token && clientProvidedConfig.server.token.trim().length > 0) {
      if (config.server) {
        config.server.token = clientProvidedConfig.server.token;
      } else {
        config.server = {
          token: clientProvidedConfig.server.token,
        };
      }
    }
    if (clientProvidedConfig?.anonymousUsageTracking?.disable !== undefined) {
      config.anonymousUsageTracking = {
        disable: clientProvidedConfig.anonymousUsageTracking.disable,
      };
    }
    return config;
  }

  private createInitClientProperties(
    clientInfo?: ClientInfo,
    clientProvidedConfig?: ClientProvidedConfig,
  ): ClientProperties {
    const clientType = this.getClientType(clientInfo);
    return {
      user: {
        [clientType]: {
          triggerMode: clientProvidedConfig?.inlineCompletion?.triggerMode,
          keybindings: clientProvidedConfig?.keybindings,
        },
      },
      session: {
        client: `${clientInfo?.name} ${clientInfo?.version ?? ""}`,
        ide: {
          name: clientInfo?.name,
          version: clientInfo?.version,
        },
        tabby_plugin: clientInfo?.tabbyPlugin ?? {
          name: `${agentName} (LSP)`,
          version: agentVersion,
        },
      },
    };
  }

  private getClientType(clientInfo?: ClientInfo | undefined | null): string {
    if (!clientInfo) {
      return "unknown";
    }
    if (clientInfo.tabbyPlugin?.name == "TabbyML.vscode-tabby") {
      return "vscode";
    } else if (clientInfo.tabbyPlugin?.name == "com.tabbyml.intellij-tabby") {
      return "intellij";
    } else if (clientInfo.tabbyPlugin?.name == "TabbyML/vim-tabby") {
      return "vim";
    }
    return clientInfo.name;
  }

  private createDataStore(): DataStore {
    const dataStore = {
      data: {},
      load: async () => {
        const params: DataStoreGetParams = { key: "data" };
        dataStore.data = await this.connection.sendRequest(DataStoreGetRequest.type, params);
      },
      save: async () => {
        const params: DataStoreSetParams = { key: "data", value: dataStore.data };
        await this.connection.sendRequest(DataStoreSetRequest.type, params);
      },
    };
    return dataStore;
  }

  private createLogger(): Logger {
    return {
      error: (msg: string, error: any) => {
        const errorMsg =
          error instanceof Error
            ? `[${error.name}] ${error.message} \n${error.stack}`
            : JSON.stringify(error, undefined, 2);
        this.connection.console.error(`${msg} ${errorMsg}`);
      },
      warn: (msg: string) => {
        this.connection.console.warn(msg);
      },
      info: (msg: string) => {
        this.connection.console.info(msg);
      },
      debug: (msg: string) => {
        this.connection.console.debug(msg);
      },
      trace: () => {},
    };
  }

  private async textDocumentPositionParamsToCompletionRequest(
    params: TextDocumentPositionParams,
    token?: CancellationToken,
  ): Promise<{ request: CompletionRequest; additionalPrefixLength?: number } | null> {
    const { textDocument, position } = params;

    this.logger.trace("Building completion context...", { uri: textDocument.uri });

    const document = this.documents.get(textDocument.uri);
    if (!document) {
      return null;
    }

    const request: CompletionRequest = {
      filepath: document.uri,
      language: document.languageId,
      text: document.getText(),
      position: document.offsetAt(position),
    };

    const notebookCell = this.notebooks.getNotebookCell(textDocument.uri);
    let additionalContext: { prefix: string; suffix: string } | undefined = undefined;
    if (notebookCell) {
      this.logger.trace("Notebook cell found:", { cell: notebookCell.kind });
      additionalContext = this.buildNotebookAdditionalContext(document, notebookCell);
    }
    if (additionalContext) {
      request.text = additionalContext.prefix + request.text + additionalContext.suffix;
      request.position += additionalContext.prefix.length;
    }

    if (this.clientCapabilities?.tabby?.editorOptions) {
      const editorOptions: EditorOptions | null = await this.connection.sendRequest(
        EditorOptionsRequest.type,
        {
          uri: params.textDocument.uri,
        },
        token,
      );
      request.indentation = editorOptions?.indentation;
    }
    if (this.clientCapabilities?.workspace) {
      const workspaceFolders = await this.connection.workspace.getWorkspaceFolders();
      request.workspace = workspaceFolders?.find((folder) => document.uri.startsWith(folder.uri))?.uri;
    }
    if (this.clientCapabilities?.tabby?.gitProvider) {
      const params: GitRepositoryParams = { uri: document.uri };
      const repo: GitRepository | null = await this.connection.sendRequest(GitRepositoryRequest.type, params, token);
      if (repo) {
        request.git = {
          root: repo.root,
          remotes: repo.remoteUrl ? [{ name: "", url: repo.remoteUrl }] : repo.remotes ?? [],
        };
      }
    } else {
      //FIXME: fallback to system `git` command
    }
    if (this.clientCapabilities?.tabby?.languageSupport) {
      request.declarations = await this.collectDeclarationSnippets(document, position, token);
    }
    if (this.recentlyChangedCodeSearch) {
      request.relevantSnippetsFromChangedFiles = await this.collectSnippetsFromRecentlyChangedFiles(document, position);
    }
    this.logger.trace("Completed completion context:", { request });
    return { request, additionalPrefixLength: additionalContext?.prefix.length };
  }

  private buildNotebookAdditionalContext(
    textDocument: TextDocument,
    notebookCell: NotebookCell,
  ): { prefix: string; suffix: string } | undefined {
    this.logger.trace("Building notebook additional context...");
    const notebook = this.notebooks.findNotebookDocumentForCell(notebookCell);
    if (!notebook) {
      return notebook;
    }
    const index = notebook.cells.indexOf(notebookCell);
    const prefix = this.buildNotebookContext(notebook, 0, index, textDocument.languageId) + "\n\n";
    const suffix =
      "\n\n" + this.buildNotebookContext(notebook, index + 1, notebook.cells.length, textDocument.languageId);

    this.logger.trace("Notebook additional context:", { prefix, suffix });
    return { prefix, suffix };
  }

  private notebookLanguageComments: { [languageId: string]: (code: string) => string } = {
    markdown: (code) => "```\n" + code + "\n```",
    python: (code) =>
      code
        .split("\n")
        .map((l) => "# " + l)
        .join("\n"),
  };

  private buildNotebookContext(notebook: NotebookDocument, from: number, to: number, languageId: string): string {
    return notebook.cells
      .slice(from, to)
      .map((cell) => {
        const textDocument = this.notebooks.getCellTextDocument(cell);
        if (!textDocument) {
          return "";
        }
        if (textDocument.languageId === languageId) {
          return textDocument.getText();
        } else if (Object.keys(this.notebookLanguageComments).includes(languageId)) {
          return this.notebookLanguageComments[languageId]?.(textDocument.getText()) ?? "";
        } else {
          return "";
        }
      })
      .join("\n\n");
  }

  private async collectDeclarationSnippets(
    textDocument: TextDocument,
    position: Position,
    token?: CancellationToken,
  ): Promise<{ filepath: string; text: string; offset?: number }[] | undefined> {
    const agentConfig = this.agent.getConfig();
    if (!agentConfig.completion.prompt.fillDeclarations.enabled) {
      return;
    }
    this.logger.debug("Collecting declaration snippets...");
    this.logger.trace("Collecting snippets for:", { textDocument: textDocument.uri, position });
    // Find symbol positions in the previous lines
    const prefixRange: Range = {
      start: { line: Math.max(0, position.line - agentConfig.completion.prompt.maxPrefixLines), character: 0 },
      end: { line: position.line, character: position.character },
    };
    const extractedSymbols = await this.extractSemanticTokenPositions(
      {
        uri: textDocument.uri,
        range: prefixRange,
      },
      token,
    );
    if (!extractedSymbols) {
      // FIXME: fallback to simple split words positions
      return undefined;
    }
    const allowedSymbolTypes = [
      "class",
      "decorator",
      "enum",
      "function",
      "interface",
      "macro",
      "method",
      "namespace",
      "struct",
      "type",
      "typeParameter",
    ];
    const symbols = extractedSymbols.filter((symbol) => allowedSymbolTypes.includes(symbol.type ?? ""));
    this.logger.trace("Found symbols in prefix text:", { symbols });

    // Loop through the symbol positions backwards
    const snippets: { filepath: string; text: string; offset?: number }[] = [];
    const snippetLocations: Location[] = [];
    for (let symbolIndex = symbols.length - 1; symbolIndex >= 0; symbolIndex--) {
      if (snippets.length >= agentConfig.completion.prompt.fillDeclarations.maxSnippets) {
        // Stop collecting snippets if the max number of snippets is reached
        break;
      }
      const symbolPosition = symbols[symbolIndex]?.position;
      if (!symbolPosition) {
        continue;
      }
      const result = await this.connection.sendRequest(
        LanguageSupportDeclarationRequest.type,
        {
          textDocument: { uri: textDocument.uri },
          position: symbolPosition,
        },
        token,
      );
      if (!result) {
        continue;
      }
      const item = Array.isArray(result) ? result[0] : result;
      if (!item) {
        continue;
      }
      const location: Location = {
        uri: "targetUri" in item ? item.targetUri : item.uri,
        range: "targetRange" in item ? item.targetRange : item.range,
      };
      this.logger.trace("Processing declaration location...", { location });
      if (location.uri == textDocument.uri && isPositionInRange(location.range.start, prefixRange)) {
        // this symbol's declaration is already contained in the prefix range
        // this also includes the case of the symbol's declaration is at this position itself
        this.logger.trace("Skipping snippet as it is contained in the prefix.");
        continue;
      }
      if (
        snippetLocations.find(
          (collectedLocation) =>
            location.uri == collectedLocation.uri && intersectionRange(location.range, collectedLocation.range),
        )
      ) {
        this.logger.trace("Skipping snippet as it is already collected.");
        continue;
      }
      this.logger.trace("Prepare to fetch text content...");
      let text: string | undefined = undefined;
      const targetDocument = this.documents.get(location.uri);
      if (targetDocument) {
        this.logger.trace("Fetching text content from synced text document.", {
          uri: targetDocument.uri,
          range: location.range,
        });
        text = targetDocument.getText(location.range);
        this.logger.trace("Fetched text content from synced text document.", { text });
      } else if (this.clientCapabilities?.tabby?.workspaceFileSystem) {
        const params: ReadFileParams = {
          uri: location.uri,
          format: "text",
          range: {
            start: { line: location.range.start.line, character: 0 },
            end: { line: location.range.end.line, character: location.range.end.character },
          },
        };
        this.logger.trace("Fetching text content from ReadFileRequest.", { params });
        const result = await this.connection.sendRequest(ReadFileRequest.type, params, token);
        this.logger.trace("Fetched text content from ReadFileRequest.", { result });
        text = result?.text;
      } else {
        // FIXME: fallback to fs
      }
      if (!text) {
        this.logger.trace("Cannot fetch text content, continue to next.", { result });
        continue;
      }
      const maxChars = agentConfig.completion.prompt.fillDeclarations.maxCharsPerSnippet;
      if (text.length > maxChars) {
        // crop the text to fit within the chars limit
        text = text.slice(0, maxChars);
        const lastNewLine = text.lastIndexOf("\n");
        if (lastNewLine > 0) {
          text = text.slice(0, lastNewLine + 1);
        }
      }
      if (text.length > 0) {
        this.logger.trace("Collected declaration snippet:", { text });
        snippets.push({ filepath: location.uri, offset: targetDocument?.offsetAt(position), text });
        snippetLocations.push(location);
      }
    }
    this.logger.debug("Completed collecting declaration snippets.");
    this.logger.trace("Collected snippets:", snippets);
    return snippets;
  }

  private async extractSemanticTokenPositions(
    location: Location,
    token?: CancellationToken,
  ): Promise<
    | {
        position: Position;
        type: string | undefined;
      }[]
    | undefined
  > {
    const result = await this.connection.sendRequest(
      LanguageSupportSemanticTokensRangeRequest.type,
      {
        textDocument: { uri: location.uri },
        range: location.range,
      },
      token,
    );
    if (!result || !result.legend || !result.legend.tokenTypes || !result.tokens || !result.tokens.data) {
      return undefined;
    }
    const { legend, tokens } = result;
    const data: number[] = Array.isArray(tokens.data) ? tokens.data : Object.values(tokens.data);
    const semanticSymbols: {
      position: Position;
      type: string | undefined;
    }[] = [];
    let line = 0;
    let character = 0;
    for (let i = 0; i + 4 < data.length; i += 5) {
      const deltaLine = data[i];
      const deltaChar = data[i + 1];
      // i + 2 is token length, not used here
      const typeIndex = data[i + 3];
      // i + 4 is type modifiers, not used here
      if (deltaLine === undefined || deltaChar === undefined || typeIndex === undefined) {
        break;
      }

      line += deltaLine;
      if (deltaLine > 0) {
        character = deltaChar;
      } else {
        character += deltaChar;
      }
      semanticSymbols.push({
        position: { line, character },
        type: legend.tokenTypes[typeIndex],
      });
    }
    return semanticSymbols;
  }

  private async collectSnippetsFromRecentlyChangedFiles(
    textDocument: TextDocument,
    position: Position,
  ): Promise<{ filepath: string; offset: number; text: string; score: number }[] | undefined> {
    const agentConfig = this.agent.getConfig();
    if (
      !agentConfig.completion.prompt.collectSnippetsFromRecentChangedFiles.enabled ||
      !this.recentlyChangedCodeSearch
    ) {
      return undefined;
    }
    this.logger.debug("Collecting snippets from recently changed files...");
    this.logger.trace("Collecting snippets for:", { document: textDocument.uri, position });
    const prefixRange: Range = {
      start: { line: Math.max(0, position.line - agentConfig.completion.prompt.maxPrefixLines), character: 0 },
      end: { line: position.line, character: position.character },
    };
    const prefixText = textDocument.getText(prefixRange);
    const query = extractNonReservedWordList(prefixText);
    const snippets = await this.recentlyChangedCodeSearch.collectRelevantSnippets(
      query,
      textDocument,
      agentConfig.completion.prompt.collectSnippetsFromRecentChangedFiles.maxSnippets,
    );
    this.logger.debug("Completed collecting snippets from recently changed files.");
    this.logger.trace("Collected snippets:", snippets);
    return snippets;
  }

  private async completionParamsToCompletionRequest(
    params: CompletionParams,
    token?: CancellationToken,
  ): Promise<{ request: CompletionRequest; additionalPrefixLength?: number } | null> {
    const result = await this.textDocumentPositionParamsToCompletionRequest(params, token);
    if (!result) {
      return null;
    }
    result.request.manually = params.context?.triggerKind === CompletionTriggerKind.Invoked;
    return result;
  }

  private async inlineCompletionParamsToCompletionRequest(
    params: InlineCompletionParams,
    token?: CancellationToken,
  ): Promise<{ request: CompletionRequest; additionalPrefixLength?: number } | null> {
    const result = await this.textDocumentPositionParamsToCompletionRequest(params, token);
    if (!result) {
      return null;
    }
    result.request.manually = params.context?.triggerKind === InlineCompletionTriggerKind.Invoked;
    return result;
  }

  private toCompletionList(
    response: CompletionResponse,
    documentPosition: TextDocumentPositionParams,
    additionalPrefixLength: number = 0,
  ): CompletionList | null {
    const { textDocument, position } = documentPosition;
    const document = this.documents.get(textDocument.uri);
    if (!document) {
      return null;
    }

    // Get word prefix if cursor is at end of a word
    const linePrefix = document.getText({
      start: { line: position.line, character: 0 },
      end: position,
    });
    const wordPrefix = linePrefix.match(/(\w+)$/)?.[0] ?? "";

    return {
      isIncomplete: response.isIncomplete,
      items: response.items.map((item): CompletionItem => {
        const insertionText = item.insertText.slice(
          document.offsetAt(position) - (item.range.start - additionalPrefixLength),
        );

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
              end: document.positionAt(item.range.end - additionalPrefixLength),
            },
          },
          data: item.data,
        };
      }),
    };
  }

  private toInlineCompletionList(
    response: CompletionResponse,
    documentPosition: TextDocumentPositionParams,
    additionalPrefixLength: number = 0,
  ): InlineCompletionList | null {
    const { textDocument } = documentPosition;
    const document = this.documents.get(textDocument.uri);
    if (!document) {
      return null;
    }

    return {
      isIncomplete: response.isIncomplete,
      items: response.items.map((item): InlineCompletionItem => {
        return {
          insertText: item.insertText,
          range: {
            start: document.positionAt(item.range.start - additionalPrefixLength),
            end: document.positionAt(item.range.end - additionalPrefixLength),
          },
          data: item.data,
        };
      }),
    };
  }

  private buildHelpMessage(issueDetail: AgentIssue, format?: "markdown" | "html"): string | undefined {
    const outputFormat = format ?? "markdown";

    // "connectionFailed"
    if (issueDetail.name == "connectionFailed") {
      if (outputFormat == "html") {
        return issueDetail.message?.replace(/\n/g, "<br/>");
      } else {
        return issueDetail.message;
      }
    }

    // "slowCompletionResponseTime" or "highCompletionTimeoutRate"
    let statsMessage = "";
    if (issueDetail.name == "slowCompletionResponseTime") {
      const stats = issueDetail.completionResponseStats;
      if (stats && stats["responses"] && stats["averageResponseTime"]) {
        statsMessage = `The average response time of recent ${stats["responses"]} completion requests is ${Number(
          stats["averageResponseTime"],
        ).toFixed(0)}ms.<br/><br/>`;
      }
    }

    if (issueDetail.name == "highCompletionTimeoutRate") {
      const stats = issueDetail.completionResponseStats;
      if (stats && stats["total"] && stats["timeouts"]) {
        statsMessage = `${stats["timeouts"]} of ${stats["total"]} completion requests timed out.<br/><br/>`;
      }
    }

    let helpMessageForRunningLargeModelOnCPU = "";
    const serverHealthState = this.agent.getServerHealthState();
    if (serverHealthState?.device === "cpu" && serverHealthState?.model?.match(/[0-9.]+B$/)) {
      helpMessageForRunningLargeModelOnCPU +=
        `Your Tabby server is running model <i>${serverHealthState?.model}</i> on CPU. ` +
        "This model may be performing poorly due to its large parameter size, please consider trying smaller models or switch to GPU. " +
        "You can find a list of recommend models in the <a href='https://tabby.tabbyml.com/'>online documentation</a>.<br/>";
    }
    let commonHelpMessage = "";
    if (helpMessageForRunningLargeModelOnCPU.length == 0) {
      commonHelpMessage += `<li>The running model ${
        serverHealthState?.model ?? ""
      } may be performing poorly due to its large parameter size. `;
      commonHelpMessage +=
        "Please consider trying smaller models. You can find a list of recommend models in the <a href='https://tabby.tabbyml.com/'>online documentation</a>.</li>";
    }
    const host = new URL(this.serverInfo?.config.endpoint ?? "http://localhost:8080").host;
    if (!(host.startsWith("localhost") || host.startsWith("127.0.0.1") || host.startsWith("0.0.0.0"))) {
      commonHelpMessage += "<li>A poor network connection. Please check your network and proxy settings.</li>";
      commonHelpMessage += "<li>Server overload. Please contact your Tabby server administrator for assistance.</li>";
    }
    let helpMessage = "";
    if (helpMessageForRunningLargeModelOnCPU.length > 0) {
      helpMessage += helpMessageForRunningLargeModelOnCPU + "<br/>";
      if (commonHelpMessage.length > 0) {
        helpMessage += "Other possible causes of this issue: <br/><ul>" + commonHelpMessage + "</ul>";
      }
    } else {
      // commonHelpMessage should not be empty here
      helpMessage += "Possible causes of this issue: <br/><ul>" + commonHelpMessage + "</ul>";
    }

    if (outputFormat == "html") {
      return statsMessage + helpMessage;
    } else {
      return (statsMessage + helpMessage)
        .replace("<br/>", " \n")
        .replace(/<i>(.*?)<\/i>/g, "$1")
        .replace(/<a[^>]*>(.*?)<\/a>/g, "$1")
        .replace(/<ul[^>]*>(.*?)<\/ul>/g, "$1")
        .replace(/<li[^>]*>(.*?)<\/li>/g, "- $1 \n");
    }
  }
}
