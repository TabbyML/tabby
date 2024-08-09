import { env, version, ExtensionContext, LogOutputChannel, LogLevel } from "vscode";
import { BaseLanguageClient, StaticFeature, FeatureState, Trace } from "vscode-languageclient";
import { InitializeParams } from "tabby-agent";
import { Config } from "../Config";

export class InitializationFeature implements StaticFeature {
  constructor(
    private readonly context: ExtensionContext,
    private readonly client: BaseLanguageClient,
    private readonly config: Config,
    private readonly logger: LogOutputChannel,
  ) {}

  getState(): FeatureState {
    return { kind: "static" };
  }

  fillInitializeParams(params: InitializeParams) {
    params.initializationOptions = {
      ...params.initializationOptions,
      config: this.config.buildClientProvidedConfig(),
      clientInfo: {
        name: `${env.appName} ${env.appHost}`,
        version: version,
        tabbyPlugin: {
          name: this.context.extension.id,
          version: this.context.extension.packageJSON.version,
        },
      },
      clientCapabilities: {
        textDocument: {
          completion: false,
        },
      },
    };
    params.trace = this.getCurrentTraceValue();
  }

  fillClientCapabilities(): void {
    // nothing
  }

  preInitialize(): void {
    // nothing
  }

  initialize(): void {
    // Sync trace setting
    this.client.setTrace(Trace.fromString(this.getCurrentTraceValue()));
    this.context.subscriptions.push(
      this.logger.onDidChangeLogLevel(async () => {
        await this.client.setTrace(Trace.fromString(this.getCurrentTraceValue()));
      }),
    );
  }

  clear(): void {
    // nothing
  }

  private getCurrentTraceValue(): "verbose" | "off" {
    const level = this.logger.logLevel;
    if (level === LogLevel.Trace) {
      return "verbose";
    } else {
      return "off";
    }
  }
}
