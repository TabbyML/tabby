import { BaseLanguageClient, StaticFeature, FeatureState } from "vscode-languageclient";
import { TelemetryEventNotification, EventParams } from "tabby-agent";

export class TelemetryFeature implements StaticFeature {
  constructor(private readonly client: BaseLanguageClient) {}

  getState(): FeatureState {
    return { kind: "static" };
  }

  fillInitializeParams() {
    // nothing
  }

  fillClientCapabilities(): void {
    // nothing
  }

  preInitialize(): void {
    // nothing
  }

  initialize(): void {
    // nothing
  }

  clear(): void {
    // nothing
  }

  async postEvent(params: EventParams): Promise<void> {
    return this.client.sendNotification(TelemetryEventNotification.method, params);
  }
}
