import { EventEmitter } from "events";
import type { Connection, Disposable } from "vscode-languageserver";
import type { ServerCapabilities } from "../protocol";
import type { Feature } from "../feature";
import type { TabbyApiClient } from "../http/tabbyApiClient";
import { ChatFeatures } from "../protocol";

export class ChatFeature extends EventEmitter implements Feature {
  private isApiAvailable = false;
  private featureRegistration: Disposable | undefined = undefined;

  constructor(readonly tabbyApiClient: TabbyApiClient) {
    super();
  }

  isAvailable(): boolean {
    return this.isApiAvailable;
  }

  private updateIsAvailable() {
    const health = this.tabbyApiClient.getServerHealth();
    const isAvailable = !!(health && health["chat_model"]);
    if (this.isApiAvailable != isAvailable) {
      this.isApiAvailable = isAvailable;
      this.emit("isAvailableUpdated", isAvailable);
    }
  }

  initialize(connection: Connection): ServerCapabilities {
    this.updateIsAvailable();
    this.tabbyApiClient.on("statusUpdated", async () => {
      this.updateIsAvailable();
      await this.syncFeatureRegistration(connection);
    });

    return {};
  }

  async initialized(connection: Connection) {
    await this.syncFeatureRegistration(connection);
  }

  private async syncFeatureRegistration(connection: Connection) {
    if (this.isApiAvailable) {
      if (!this.featureRegistration) {
        try {
          this.featureRegistration = await connection.client.register(ChatFeatures.type);
        } catch (error) {
          // client may not support feature registration, ignore this error
        }
      }
    } else {
      this.featureRegistration?.dispose();
      this.featureRegistration = undefined;
    }
  }
}
