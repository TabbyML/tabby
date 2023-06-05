import { workspace } from "vscode";
import { TabbyAgent, AgentConfig } from "tabby-agent";

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
    super.initialize({
      config: this.getWorkspaceConfiguration()
    });

    workspace.onDidChangeConfiguration((event) => {
      if (event.affectsConfiguration("tabby")) {
        const config = this.getWorkspaceConfiguration();
        super.updateConfig(config);
      }
    });
  }

  private getWorkspaceConfiguration(): AgentConfig {
    const configuration = workspace.getConfiguration("tabby");
    const serverUrl = configuration.get<string>("serverUrl");
    return {
      server: {
        endpoint: serverUrl
      }
    }
  }
}
