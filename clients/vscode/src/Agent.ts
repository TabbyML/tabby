import { workspace } from "vscode";
import { TabbyAgent } from "tabby-agent";

export class Agent extends TabbyAgent {
  private static instance: Agent;
  static getInstance(): Agent {
    if (!Agent.instance) {
      Agent.instance = new Agent();
    }
    return Agent.instance;
  }

  private constructor() {
    super();
    this.updateConfiguration();
    workspace.onDidChangeConfiguration((event) => {
      if (event.affectsConfiguration("tabby")) {
        this.updateConfiguration();
      }
    });
  }

  private updateConfiguration() {
    const configuration = workspace.getConfiguration("tabby");
    const serverUrl = configuration.get("serverUrl", "http://localhost:5000");
    this.setServerUrl(serverUrl);
  }
}
