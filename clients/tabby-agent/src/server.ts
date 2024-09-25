import type { Feature } from "./feature";
import { createConnection as nodeCreateConnection } from "vscode-languageserver/node";
import {
  createConnection as browserCreateConnection,
  BrowserMessageReader,
  BrowserMessageWriter,
} from "vscode-languageserver/browser";
import { ProposedFeatures, TextDocumentSyncKind, NotebookDocuments } from "vscode-languageserver";
import {
  InitializeParams,
  InitializeResult,
  ClientInfo,
  ClientCapabilities,
  ServerCapabilities,
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
} from "./protocol";
import { TextDocuments } from "./lsp/textDocuments";
import { TextDocument } from "vscode-languageserver-textdocument";
import { deepmerge } from "deepmerge-ts";
import { isBrowser } from "./env";
import { getLogger, LoggerManager } from "./logger";
import { DataStore } from "./dataStore";
import { Configurations } from "./config";
import { CertsLoader } from "./certsLoader";
import { AnonymousUsageLogger } from "./telemetry";
import { TabbyApiClient } from "./http/tabbyApiClient";
import { GitContextProvider } from "./git";
import { RecentlyChangedCodeSearch } from "./codeSearch/recentlyChanged";
import { CodeLensProvider } from "./codeLens";
import { CompletionProvider } from "./codeCompletion";
import { ChatFeature } from "./chat";
import { ChatEditProvider } from "./chat/inlineEdit";
import { CommitMessageGenerator } from "./chat/generateCommitMessage";
import { StatusProvider } from "./status";
import { CommandProvider } from "./command";
import { name as serverName, version as serverVersion } from "../package.json";
import "./utils/array";

export class Server {
  private readonly logger = getLogger("TabbyLSP");
  private readonly connection = isBrowser
    ? browserCreateConnection(ProposedFeatures.all, new BrowserMessageReader(self), new BrowserMessageWriter(self))
    : nodeCreateConnection(ProposedFeatures.all);

  private readonly documents = new TextDocuments(TextDocument);
  private readonly notebooks = new NotebookDocuments(this.documents);

  private readonly dataStore = new DataStore();
  private readonly configurations = new Configurations(this.dataStore);

  private readonly certsLoader = new CertsLoader(this.configurations);
  private readonly anonymousUsageLogger = new AnonymousUsageLogger(this.dataStore, this.configurations);
  private readonly tabbyApiClient = new TabbyApiClient(this.configurations, this.anonymousUsageLogger);

  private readonly gitContextProvider = new GitContextProvider();
  private readonly recentlyChangedCodeSearch = new RecentlyChangedCodeSearch(this.configurations, this.documents);

  private readonly codeLensProvider = new CodeLensProvider(this.documents);
  private readonly completionProvider = new CompletionProvider(
    this.configurations,
    this.tabbyApiClient,
    this.documents,
    this.notebooks,
    this.anonymousUsageLogger,
    this.gitContextProvider,
    this.recentlyChangedCodeSearch,
  );
  private readonly chatFeature = new ChatFeature(this.tabbyApiClient);
  private readonly chatEditProvider = new ChatEditProvider(this.configurations, this.tabbyApiClient, this.documents);
  private readonly commitMessageGenerator = new CommitMessageGenerator(
    this.configurations,
    this.tabbyApiClient,
    this.gitContextProvider,
  );

  private readonly statusProvider = new StatusProvider(this.dataStore, this.configurations, this.tabbyApiClient);
  private readonly commandProvider = new CommandProvider(this.chatEditProvider, this.statusProvider);

  private clientCapabilities: ClientCapabilities | undefined;

  async listen() {
    await this.preInitialize();
    this.documents.listen(this.connection);
    this.notebooks.listen(this.connection);
    this.connection.listen();
  }

  private async preInitialize() {
    // pre-initialize components
    const loggerManager = LoggerManager.getInstance();
    loggerManager.preInitialize(this.configurations);
    loggerManager.attachLspConnection(this.connection);

    await this.dataStore.preInitialize();
    await this.configurations.preInitialize();
    await this.certsLoader.preInitialize();

    // Lifecycle methods
    this.connection.onInitialize(async (params) => {
      return this.initialize(params);
    });
    this.connection.onInitialized(async () => {
      return this.initialized();
    });
    this.connection.onShutdown(async () => {
      return this.shutdown();
    });
    this.connection.onExit(async () => {
      return this.exit();
    });

    // FIXME(@icycodes): remove deprecated methods
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
  }

  private async initialize(params: InitializeParams): Promise<InitializeResult> {
    this.logger.info("Initializing...");
    const clientInfo: ClientInfo | undefined = deepmerge(
      params.clientInfo,
      params.initializationOptions?.clientInfo ?? {},
    );
    const clientCapabilities: ClientCapabilities = deepmerge(
      params.capabilities,
      params.initializationOptions?.clientCapabilities ?? {},
    );
    this.clientCapabilities = clientCapabilities;

    const clientProvidedConfig: ClientProvidedConfig = params.initializationOptions?.config ?? {};

    const baseCapabilities: ServerCapabilities = {
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
    };

    this.logger.debug("Initializing internal components...");
    await this.dataStore.initialize(this.connection, clientCapabilities);
    await this.configurations.initialize(this.connection, clientCapabilities, clientProvidedConfig);
    await this.anonymousUsageLogger.initialize(clientInfo);
    await this.tabbyApiClient.initialize(clientInfo);
    this.logger.debug("Internal components initialized.");

    this.logger.debug("Initializing feature components...");
    const capabilities: ServerCapabilities[] = await [
      this.gitContextProvider,
      this.recentlyChangedCodeSearch,
      this.codeLensProvider,
      this.completionProvider,
      this.chatFeature,
      this.chatEditProvider,
      this.commitMessageGenerator,
      this.statusProvider,
      this.commandProvider,
    ].mapAsync((feature: Feature) => {
      return feature.initialize(this.connection, clientCapabilities, clientProvidedConfig);
    });
    this.logger.debug("Feature components initialized.");

    const serverCapabilities: ServerCapabilities = deepmerge(baseCapabilities, ...capabilities);

    const result: InitializeResult = {
      capabilities: serverCapabilities,
      serverInfo: {
        name: serverName,
        version: serverVersion,
      },
    };

    this.logger.info("Initialize done.");
    this.anonymousUsageLogger.uniqueEvent("AgentInitialized"); // telemetry event, no wait
    return result;
  }

  private async initialized(): Promise<void> {
    this.logger.info("Received initialized notification.");
    await [this.completionProvider, this.chatFeature].mapAsync((feature: Feature) => {
      return feature.initialized?.(this.connection);
    });

    // FIXME(@icycodes): remove deprecated methods
    if (this.clientCapabilities?.tabby?.agent) {
      this.tabbyApiClient.on("statusUpdated", async () => {
        this.connection.sendNotification(AgentServerInfoSync.type, { serverInfo: this.buildServerInfo() });

        this.connection.sendNotification(AgentStatusSync.type, { status: this.buildAgentStatus() });

        this.connection.sendNotification(AgentIssuesSync.type, { issues: this.buildAgentIssues().issues });
      });

      this.tabbyApiClient.on("isConnectingUpdated", async () => {
        this.connection.sendNotification(AgentStatusSync.type, { status: this.buildAgentStatus() });
      });

      this.tabbyApiClient.on("hasCompletionResponseTimeIssueUpdated", async () => {
        this.connection.sendNotification(AgentIssuesSync.type, { issues: this.buildAgentIssues().issues });
      });
    }
  }

  private async shutdown() {
    this.logger.info("Shutting down...");
    await [this.recentlyChangedCodeSearch, this.completionProvider].mapAsync((feature: Feature) => {
      return feature.shutdown?.();
    });
    await this.tabbyApiClient.shutdown();
    this.logger.info("Shutdown done.");
  }

  private exit() {
    return process.exit(0);
  }

  // FIXME(@icycodes): remove adapters for deprecated methods
  // adapters for deprecated methods
  private async getServerInfo(): Promise<ServerInfo> {
    return this.buildServerInfo();
  }

  private async getStatus(): Promise<Status> {
    return this.buildAgentStatus();
  }

  private async getIssues(): Promise<IssueList> {
    return this.buildAgentIssues();
  }

  private async getIssueDetail(params: IssueDetailParams): Promise<IssueDetailResult | null> {
    if (params.name && this.tabbyApiClient.hasHelpMessage()) {
      return {
        name: params.name,
        helpMessage: this.tabbyApiClient.getHelpMessage(params.helpMessageFormat),
      };
    }
    return null;
  }

  private buildServerInfo(): ServerInfo {
    return {
      config: this.configurations.getMergedConfig().server,
      health: this.tabbyApiClient.getServerHealth() || null,
    };
  }

  private buildAgentStatus(): Status {
    let agentStatus: Status = "notInitialized";
    switch (this.tabbyApiClient.getStatus()) {
      case "noConnection":
        agentStatus = "disconnected";
        break;
      case "unauthorized":
        agentStatus = "unauthorized";
        break;
      case "ready":
        agentStatus = "ready";
        break;
    }

    if (this.tabbyApiClient.isConnecting()) {
      agentStatus = "notInitialized";
    }
    return agentStatus;
  }

  private buildAgentIssues(): IssueList {
    if (this.tabbyApiClient.getStatus() === "noConnection") {
      return { issues: ["connectionFailed"] };
    } else if (this.tabbyApiClient.hasCompletionResponseTimeIssue()) {
      return { issues: ["slowCompletionResponseTime"] };
    } else {
      return { issues: [] };
    }
  }
  // end of adapters for deprecated methods
}
