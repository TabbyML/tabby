import { EventEmitter } from "events";
import { Connection } from "vscode-languageserver";
import { ClientCapabilities, ServerCapabilities, Config, ConfigRequest, ConfigDidChangeNotification } from "./protocol";
import type { Feature } from "./feature";
import { TabbyAgent } from "../TabbyAgent";

export class ConfigProvider extends EventEmitter implements Feature {
  constructor(private readonly agent: TabbyAgent) {
    super();
    this.agent.on("configUpdated", async () => {
      this.update();
    });
  }

  private async update() {
    const status = await this.getConfig();
    this.emit("updated", status);
  }

  setup(connection: Connection, clientCapabilities: ClientCapabilities): ServerCapabilities {
    connection.onRequest(ConfigRequest.type, async () => {
      return this.getConfig();
    });
    if (clientCapabilities.tabby?.configDidChangeListener) {
      this.on("updated", (config: Config) => {
        connection.sendNotification(ConfigDidChangeNotification.type, config);
      });
    }
    return {};
  }

  async getConfig(): Promise<Config> {
    const agentConfig = this.agent.getConfig();
    return {
      server: agentConfig.server,
    };
  }
}
