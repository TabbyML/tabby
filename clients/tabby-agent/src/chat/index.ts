import type { Connection } from "vscode-languageserver";
import type { ServerCapabilities } from "../protocol";
import type { Feature } from "../feature";
import type { TabbyApiClient } from "../http/tabbyApiClient";
import { RegistrationRequest, UnregistrationRequest } from "vscode-languageserver";
import { ChatFeatureRegistration } from "../protocol";

export class ChatFeature implements Feature {
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
    if (this.tabbyApiClient.isChatApiAvailable()) {
      connection.sendRequest(RegistrationRequest.type, {
        registrations: [
          {
            id: ChatFeatureRegistration.type.method,
            method: ChatFeatureRegistration.type.method,
          },
        ],
      });
    } else {
      connection.sendRequest(UnregistrationRequest.type, {
        unregisterations: [
          {
            id: ChatFeatureRegistration.type.method,
            method: ChatFeatureRegistration.type.method,
          },
        ],
      });
    }
  }
}
