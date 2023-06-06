import { workspace, env, version, UIKind } from "vscode";
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
    const uiKind = Object.keys(UIKind)[Object.values(UIKind).indexOf(env.uiKind)];
    super.initialize({
      config: this.getWorkspaceConfiguration(),
      client: `VSCode ${uiKind} ${version}`,
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
    const agentLogs = configuration.get<"debug" | "error" | "silent">("agentLogs");
    return {
      server: {
        endpoint: serverUrl,
      },
      logs: {
        level: agentLogs,
      },
    };
  }
}
