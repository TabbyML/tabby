import { env, ExtensionContext } from "vscode";
import { BaseLanguageClient, StaticFeature, FeatureState, Disposable } from "vscode-languageclient";
import {
  ClientCapabilities,
  DataStoreGetRequest,
  DataStoreGetParams,
  DataStoreSetRequest,
  DataStoreSetParams,
} from "tabby-agent";

export class DataStoreFeature implements StaticFeature {
  private disposables: Disposable[] = [];

  constructor(
    private readonly context: ExtensionContext,
    private readonly client: BaseLanguageClient,
  ) {}

  getState(): FeatureState {
    return { kind: "static" };
  }

  fillInitializeParams() {
    // nothing
  }

  fillClientCapabilities(capabilities: ClientCapabilities): void {
    if (env.appHost === "desktop") {
      return;
    }
    capabilities.tabby = {
      ...capabilities.tabby,
      dataStore: true,
    };
  }

  preInitialize(): void {
    // nothing
  }

  initialize(): void {
    if (env.appHost === "desktop") {
      return;
    }
    this.disposables.push(
      this.client.onRequest(DataStoreGetRequest.type, (params: DataStoreGetParams) => {
        const data: Record<string, unknown> = this.context.globalState.get("data", {});
        return data[params.key];
      }),
    );
    this.disposables.push(
      this.client.onRequest(DataStoreSetRequest.type, async (params: DataStoreSetParams) => {
        const data: Record<string, unknown> = this.context.globalState.get("data", {});
        data[params.key] = params.value;
        this.context.globalState.update("data", data);
        return true;
      }),
    );
  }

  clear(): void {
    this.disposables.forEach((disposable) => disposable.dispose());
    this.disposables = [];
  }
}
