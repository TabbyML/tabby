import { workspace, env, version } from "vscode";
import { TabbyAgent, AgentConfig } from "tabby-agent";

function getWorkspaceConfiguration(): Partial<AgentConfig> {
  const configuration = workspace.getConfiguration("tabby");
  const config: Partial<AgentConfig> = {};
  const serverUrl = configuration.get<string>("serverUrl");
  if (serverUrl) {
    config.server = {
      endpoint: serverUrl,
    };
  }
  const agentLogs = configuration.get<"debug" | "error" | "silent">("agentLogs");
  if (agentLogs) {
    config.logs = {
      level: agentLogs,
    };
  }
  return config;
}

var instance: TabbyAgent;

export function agent(): TabbyAgent {
  if (!instance) {
    throw new Error("Tabby Agent not initialized");
  }
  return instance;
}

export async function createAgentInstance(): Promise<TabbyAgent> {
  if (!instance) {
    const agent = await TabbyAgent.create();
    agent.initialize({
      config: getWorkspaceConfiguration(),
      client: `${env.appName} ${env.appHost} ${version}`,
    });
    workspace.onDidChangeConfiguration((event) => {
      if (event.affectsConfiguration("tabby")) {
        const config = getWorkspaceConfiguration();
        agent.updateConfig(config);
      }
    });
    instance = agent;
  }
  return instance;
}
