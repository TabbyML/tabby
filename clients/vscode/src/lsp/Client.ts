import { CodeActionProvider, ExtensionContext, languages } from "vscode";
import { BaseLanguageClient } from "vscode-languageclient";
import { AgentFeature } from "./AgentFeature";
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
import { getLogger } from "../logger";

export class Client {
  private readonly logger = getLogger("");
  readonly agent: AgentFeature;
  readonly chat: ChatFeature;
  readonly telemetry: TelemetryFeature;
  constructor(
    private readonly context: ExtensionContext,
    readonly languageClient: BaseLanguageClient,
  ) {
    this.agent = new AgentFeature(this.languageClient);
    this.chat = new ChatFeature(this.languageClient);
    this.telemetry = new TelemetryFeature(this.languageClient);
    this.languageClient.registerFeature(this.agent);
    this.languageClient.registerFeature(this.chat);
    this.languageClient.registerFeature(this.telemetry);
    this.languageClient.registerFeature(new DataStoreFeature(this.context, this.languageClient));
    this.languageClient.registerFeature(new EditorOptionsFeature(this.languageClient));
    this.languageClient.registerFeature(new LanguageSupportFeature(this.languageClient));
    this.languageClient.registerFeature(new WorkspaceFileSystemFeature(this.languageClient));

    const codeLensMiddleware = new CodeLensMiddleware();
    this.languageClient.middleware.provideCodeLenses = (document, token, next) =>
      codeLensMiddleware.provideCodeLenses(document, token, next);
  }

  async start(): Promise<void> {
    await this.languageClient.start();
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
