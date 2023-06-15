import { ExtensionContext, workspace, env, version } from "vscode";
import { TabbyAgent, AgentConfig, DataStore } from "tabby-agent";

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

export async function createAgentInstance(context: ExtensionContext): Promise<TabbyAgent> {
  if (!instance) {
    // NOTE: env.appHost will be "desktop" when running target "Run Web Extension in VS Code"
    // To test data store in the web extension, run target "Run Web Extension in Browser"
    const extensionDataStore: DataStore = {
      data: {},
      load: async function () {
        this.data = context.globalState.get("data", {});
      },
      save: async function () {
        context.globalState.update("data", this.data);
      },
    };
    const agent = await TabbyAgent.create({ dataStore: env.appHost === "desktop" ? undefined : extensionDataStore });
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
