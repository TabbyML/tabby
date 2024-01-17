import { ExtensionContext, workspace, env, version } from "vscode";
import { TabbyAgent, AgentInitOptions, PartialAgentConfig, ClientProperties, DataStore } from "tabby-agent";

function buildInitOptions(context: ExtensionContext): AgentInitOptions {
  const configuration = workspace.getConfiguration("rumicode");
  const config: PartialAgentConfig = {};
  const endpoint = configuration.get<string>("api.endpoint");
  const token = configuration.get<string>("api.token");

  config.server = {};
  if (endpoint && endpoint.trim().length > 0) {
    config.server.endpoint = endpoint.trim();
  }

  if (token && token.trim().length > 0) {
    config.server.token = token.trim();
  }

  const clientProperties: ClientProperties = {
    user: {
      vscode: {
        triggerMode: configuration.get("inlineCompletion.triggerMode", "automatic"),
        keybindings: configuration.get("keybindings", "vscode-style"),
      },
    },
    session: {
      client: `${env.appName} ${env.appHost} ${version}, ${context.extension.id} ${context.extension.packageJSON.version}`,
      ide: {
        name: `${env.appName} ${env.appHost}`,
        version: version,
      },
      tabby_plugin: {
        name: context.extension.id,
        version: context.extension.packageJSON.version,
      },
    },
  };
  const extensionDataStore: DataStore = {
    data: {},
    load: async function () {
      this.data = context.globalState.get("data", {});
    },
    save: async function () {
      context.globalState.update("data", this.data);
    },
  };
  const dataStore = env.appHost === "desktop" ? undefined : extensionDataStore;
  return { config, clientProperties, dataStore };
}

let instance: TabbyAgent | undefined = undefined;

export function agent(): TabbyAgent {
  if (!instance) {
    throw new Error("RumiCode not initialized");
  }
  return instance;
}

export async function createAgentInstance(context: ExtensionContext): Promise<TabbyAgent> {
  if (!instance) {
    const agent = new TabbyAgent();
    const initPromise = agent.initialize(buildInitOptions(context));
    workspace.onDidChangeConfiguration(async (event) => {
      await initPromise;
      const configuration = workspace.getConfiguration("rumicode");
      if (event.affectsConfiguration("rumicode.api.endpoint")) {
        const endpoint = configuration.get<string>("api.endpoint");
        if (endpoint && endpoint.trim().length > 0) {
          agent.updateConfig("server.endpoint", endpoint);
        } else {
          agent.clearConfig("server.endpoint");
        }
      }
      if (event.affectsConfiguration("rumicode.api.token")) {
        const token = configuration.get<string>("api.token");
        if (token && token.trim().length > 0) {
          agent.updateConfig("server.token", token);
        } else {
          agent.clearConfig("server.token");
        }
      }

      if (event.affectsConfiguration("rumicode.inlineCompletion.triggerMode")) {
        const triggerMode = configuration.get<string>("inlineCompletion.triggerMode", "automatic");
        agent.updateClientProperties("user", "vscode.triggerMode", triggerMode);
      }
      if (event.affectsConfiguration("rumicode.keybindings")) {
        const keybindings = configuration.get<string>("keybindings", "vscode-style");
        agent.updateClientProperties("user", "vscode.keybindings", keybindings);
      }
    });
    instance = agent;
  }
  return instance;
}

export async function disposeAgentInstance(): Promise<void> {
  if (instance) {
    await instance.finalize();
    instance = undefined;
  }
}
