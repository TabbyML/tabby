import { EventEmitter } from "events";
import { env, ExtensionContext } from "vscode";
import { BaseLanguageClient, StaticFeature, FeatureState, Disposable } from "vscode-languageclient";
import {
  InitializeParams,
  ClientCapabilities,
  DataStoreRecords,
  DataStoreDidUpdateNotification,
  DataStoreUpdateRequest,
} from "tabby-agent";

export class DataStoreFeature extends EventEmitter implements StaticFeature {
  private disposables: Disposable[] = [];

  constructor(
    private readonly context: ExtensionContext,
    private readonly client: BaseLanguageClient,
  ) {
    super();
  }

  getState(): FeatureState {
    return { kind: "static" };
  }

  fillInitializeParams(params: InitializeParams) {
    if (env.appHost === "desktop") {
      return;
    }
    params.initializationOptions = {
      ...params.initializationOptions,
      dataStoreRecords: this.getRecords(),
    };
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
      this.client.onRequest(DataStoreUpdateRequest.type, async (params: DataStoreRecords) => {
        this.update(params);
        return true;
      }),
    );
    this.on("didUpdate", (records) => {
      this.client.sendNotification(DataStoreDidUpdateNotification.type, records);
    });
  }

  private getRecords(): DataStoreRecords {
    return this.context.globalState.get("data", {});
  }

  private update(records: DataStoreRecords) {
    this.context.globalState.update("data", records);
    this.emit("didUpdate", records);
  }

  clear(): void {
    this.disposables.forEach((disposable) => disposable.dispose());
    this.disposables = [];
  }
}
