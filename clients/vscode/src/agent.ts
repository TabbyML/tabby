import { ExtensionContext, workspace, env, version } from "vscode";
import { TabbyAgent, AgentInitOptions, PartialAgentConfig, ClientProperties, DataStore } from "tabby-agent";

function buildInitOptions(context: ExtensionContext): AgentInitOptions {
  const configuration = workspace.getConfiguration("tabby");
  const config: PartialAgentConfig = {};
  const endpoint = configuration.get<string>("api.endpoint");
  if (endpoint && endpoint.trim().length > 0) {
    config.server = {
      endpoint,
    };
  }
  const token = context.globalState.get<string>("server.token");
  if (token && token.trim().length > 0) {
    if (config.server) {
      config.server.token = token;
    } else {
      config.server = {
        token,
      };
    }
  }
  const anonymousUsageTrackingDisabled = configuration.get<boolean>("usage.anonymousUsageTracking", false);
  config.anonymousUsageTracking = {
    disable: anonymousUsageTrackingDisabled,
  };
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
    throw new Error("Tabby Agent not initialized");
  }
  return instance;
}

export async function createAgentInstance(context: ExtensionContext): Promise<TabbyAgent> {
  if (!instance) {
    const agent = new TabbyAgent();
    const initPromise = agent.initialize(buildInitOptions(context));
    workspace.onDidChangeConfiguration(async (event) => {
      await initPromise;
      const configuration = workspace.getConfiguration("tabby");
      if (event.affectsConfiguration("tabby.api.endpoint")) {
        const endpoint = configuration.get<string>("api.endpoint");
        if (endpoint && endpoint.trim().length > 0) {
          agent.updateConfig("server.endpoint", endpoint);
        } else {
          agent.clearConfig("server.endpoint");
        }
      }
      if (event.affectsConfiguration("tabby.usage.anonymousUsageTracking")) {
        const anonymousUsageTrackingDisabled = configuration.get<boolean>("usage.anonymousUsageTracking", false);
        agent.updateConfig("anonymousUsageTracking.disable", anonymousUsageTrackingDisabled);
      }
      if (event.affectsConfiguration("tabby.inlineCompletion.triggerMode")) {
        const triggerMode = configuration.get<string>("inlineCompletion.triggerMode", "automatic");
        agent.updateClientProperties("user", "vscode.triggerMode", triggerMode);
      }
      if (event.affectsConfiguration("tabby.keybindings")) {
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
