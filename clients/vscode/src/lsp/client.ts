import { CodeActionProvider, ExtensionContext, Uri, languages } from "vscode";
import { LanguageClientOptions } from "vscode-languageclient";
import { LanguageClient as NodeLanguageClient, ServerOptions, TransportKind } from "vscode-languageclient/node";
import { LanguageClient as BrowserLanguageClient } from "vscode-languageclient/browser";
import { BaseLanguageClient } from "vscode-languageclient";
import { AgentStatusFeature } from "./AgentStatusFeature";
import { AgentConfigFeature } from "./AgentConfigFeature";
import { ChatFeature } from "./ChatFeature";
import { CodeLensMiddleware } from "./CodeLensMiddleware";
import { ConfigurationMiddleware } from "./ConfigurationMiddleware";
import { ConfigurationSyncFeature } from "./ConfigurationSyncFeature";
import { DataStoreFeature } from "./DataStoreFeature";
import { EditorOptionsFeature } from "./EditorOptionsFeature";
import { GitProviderFeature } from "./GitProviderFeature";
import { InitializationFeature } from "./InitializationFeature";
import { InlineCompletionFeature } from "./InlineCompletionFeature";
import { LanguageSupportFeature } from "./LanguageSupportFeature";
import { TelemetryFeature } from "./TelemetryFeature";
import { WorkspaceFileSystemFeature } from "./WorkspaceFileSystemFeature";
import { Config } from "../Config";
import { InlineCompletionProvider } from "../InlineCompletionProvider";
import { GitProvider } from "../git/GitProvider";
import { getLogger, LogOutputChannel } from "../logger";
import { WorkSpaceFeature } from "./WorkspaceFeature";
import { FileTrackerFeature } from "./FileTrackFeature";
import { isBrowser } from "../env";

export function createClient(context: ExtensionContext, logger: LogOutputChannel): Client {
  const clientOptions: LanguageClientOptions = {
    documentSelector: [
      { scheme: "file" },
      { scheme: "vscode-vfs" },
      { scheme: "untitled" },
      { scheme: "vscode-notebook-cell" },
      { scheme: "vscode-userdata" },
    ],
    outputChannel: logger,
  };
  if (isBrowser) {
    const workerModulePath = Uri.joinPath(context.extensionUri, "dist/tabby-agent/browser/index.mjs");
    const worker = new Worker(workerModulePath.toString());
    const languageClient = new BrowserLanguageClient("Tabby", "Tabby", clientOptions, worker);
    return new Client(context, languageClient);
  } else {
    const serverModulePath = context.asAbsolutePath("dist/tabby-agent/node/index.js");
    const serverOptions: ServerOptions = {
      run: {
        module: serverModulePath,
        transport: TransportKind.ipc,
      },
      debug: {
        module: serverModulePath,
        transport: TransportKind.ipc,
      },
    };
    const languageClient = new NodeLanguageClient("Tabby", serverOptions, clientOptions);
    return new Client(context, languageClient);
  }
}

export class Client {
  private readonly logger = getLogger("");
  readonly status: AgentStatusFeature;
  readonly agentConfig: AgentConfigFeature;
  readonly chat: ChatFeature;
  readonly telemetry: TelemetryFeature;
  readonly workspace: WorkSpaceFeature;
  readonly fileTrack: FileTrackerFeature;

  constructor(
    private readonly context: ExtensionContext,
    readonly languageClient: BaseLanguageClient,
  ) {
    this.status = new AgentStatusFeature(this.languageClient);
    this.agentConfig = new AgentConfigFeature(this.languageClient);
    this.chat = new ChatFeature(this.languageClient);
    this.workspace = new WorkSpaceFeature(this.languageClient);
    this.telemetry = new TelemetryFeature(this.languageClient);
    this.fileTrack = new FileTrackerFeature(this, this.context);
    this.languageClient.registerFeature(this.status);
    this.languageClient.registerFeature(this.agentConfig);
    this.languageClient.registerFeature(this.chat);
    this.languageClient.registerFeature(this.workspace);
    this.languageClient.registerFeature(this.telemetry);
    this.languageClient.registerFeature(this.fileTrack);
    this.languageClient.registerFeature(new DataStoreFeature(this.context, this.languageClient));
    this.languageClient.registerFeature(new EditorOptionsFeature(this.languageClient));
    this.languageClient.registerFeature(new LanguageSupportFeature(this.languageClient));
    this.languageClient.registerFeature(new WorkspaceFileSystemFeature(this.languageClient));

    const codeLensMiddleware = new CodeLensMiddleware();
    this.languageClient.middleware.provideCodeLenses = (document, token, next) =>
      codeLensMiddleware.provideCodeLenses(document, token, next);
  }

  async start(): Promise<void> {
    return this.languageClient.start();
  }

  async stop(): Promise<void> {
    return this.languageClient.stop();
  }

  registerConfigManager(config: Config): void {
    const initializationFeature = new InitializationFeature(this.context, this.languageClient, config, this.logger);
    this.languageClient.registerFeature(initializationFeature);

    const configMiddleware = new ConfigurationMiddleware(config);
    if (!this.languageClient.middleware.workspace) {
      this.languageClient.middleware.workspace = {};
    }
    this.languageClient.middleware.workspace.configuration = () => configMiddleware.configuration();

    const configSyncFeature = new ConfigurationSyncFeature(this.languageClient, config);
    this.languageClient.registerFeature(configSyncFeature);
  }

  registerInlineCompletionProvider(provider: InlineCompletionProvider): void {
    const feature = new InlineCompletionFeature(this.languageClient, provider);
    this.languageClient.registerFeature(feature);
  }

  registerGitProvider(provider: GitProvider): void {
    const feature = new GitProviderFeature(this.languageClient, provider);
    this.languageClient.registerFeature(feature);
  }

  registerCodeActionProvider(provider: CodeActionProvider) {
    this.context.subscriptions.push(languages.registerCodeActionsProvider("*", provider));
  }
}
