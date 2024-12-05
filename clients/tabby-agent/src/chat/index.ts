import type { Connection, Disposable } from "vscode-languageserver";
import type { ServerCapabilities } from "../protocol";
import type { Feature } from "../feature";
import type { TabbyApiClient } from "../http/tabbyApiClient";
import { ChatFeatures } from "../protocol";

export class ChatFeature implements Feature {
  private featureRegistration: Disposable | undefined = undefined;

  constructor(private readonly tabbyApiClient: TabbyApiClient) {}

  initialize(): ServerCapabilities {
    return {};
  }

  async initialized(connection: Connection) {
    await this.syncFeatureRegistration(connection);
    this.tabbyApiClient.on("statusUpdated", async () => {
      await this.syncFeatureRegistration(connection);
    });
  }

  private async syncFeatureRegistration(connection: Connection) {
    if (this.tabbyApiClient.isChatApiAvailable() && !this.featureRegistration) {
      this.featureRegistration = await connection.client.register(ChatFeatures.type);
    } else {
      this.featureRegistration?.dispose();
      this.featureRegistration = undefined;
    }
  }
}
